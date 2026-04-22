# run with:  python hpc_diag.py <path-to-env.yml>
# python hpc_diag.py pg/settings_10weeks_7days/env.yml > diag.txt 2>&1
# cat diag.txt

import os, sys, json
from pathlib import Path

# ---- edit if you don't want to pass as arg ----
ENV_YML_PATH = None  # e.g. "pg/settings_10weeks_7days/env.yml"

if len(sys.argv) > 1:
    ENV_YML_PATH = sys.argv[1]
assert ENV_YML_PATH, "pass env.yml path as arg or set ENV_YML_PATH"

env_yml = Path(ENV_YML_PATH).resolve()
print(f"=== env.yml ===")
print(f"path: {env_yml}  (exists={env_yml.exists()})")
print(env_yml.read_text())

# --- load env.yml ---
try:
    from ruamel.yaml import YAML
    env = YAML(typ="safe").load(env_yml.read_text())
except ImportError:
    import yaml
    env = yaml.safe_load(env_yml.read_text())

# Resolve paths relative to env.yml's parent's parent (repo root) — adjust if needed
repo_root = env_yml.parent.parent.parent  # env.yml is in pg/settings_*/env.yml
print(f"\nrepo_root (inferred): {repo_root}")
print(f"CWD:                  {Path.cwd()}")

rg_path   = (repo_root / env["RESOURCE_GROUPS"]).resolve()
prof_path = (repo_root / env["RESOURCE_GROUP_PROFILES"]).resolve()

print(f"\n=== paths ===")
print(f"RESOURCE_GROUPS:         {rg_path}  (exists={rg_path.exists()})")
print(f"RESOURCE_GROUP_PROFILES: {prof_path}  (exists={prof_path.exists()})")

# --- what's actually on disk? ---
print(f"\n=== contents of RESOURCE_GROUPS ===")
if rg_path.exists():
    for p in sorted(rg_path.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(rg_path)}")
else:
    print("  (missing!)")

print(f"\n=== contents of RESOURCE_GROUP_PROFILES ===")
if prof_path.exists():
    for p in sorted(prof_path.iterdir()):
        print(f"  {p.name}")
else:
    print("  (missing!)")

# --- inspect every JSON ---
print(f"\n=== group JSON inventory ===")
for j in sorted(rg_path.rglob("*.json")) if rg_path.exists() else []:
    try:
        d = json.loads(j.read_text())
    except Exception as e:
        print(f"  {j.relative_to(rg_path)}  ✗ parse error: {e}")
        continue
    tech = d.get("technology", "<missing>")
    existing = d.get("existing", "<default False>")
    extras = [k for k in d if k not in
              ("technology","existing","tree","metadata","profiles","site_map")]
    print(f"  {j.relative_to(rg_path)}")
    print(f"    tech={tech!r}  existing={existing}  extras={extras}")
    for key in ("metadata","profiles","site_map"):
        if key in d:
            local  = (j.parent / d[key]).exists()
            in_prof = (prof_path / d[key]).exists() if prof_path.exists() else False
            found = "JSON-dir" if local else ("PROFILES-dir" if in_prof else "NOT FOUND")
            print(f"    {key}: {d[key]}  [{found}]")

# --- simulate PG's load and the failing call ---
print(f"\n=== simulate PowerGenome build_resource_clusters ===")
os.environ["RESOURCE_GROUPS"]         = str(rg_path)
os.environ["RESOURCE_GROUP_PROFILES"] = str(prof_path)

try:
    from powergenome import params as pg_params
    pg_params.SETTINGS["RESOURCE_GROUPS"]         = str(rg_path)
    pg_params.SETTINGS["RESOURCE_GROUP_PROFILES"] = str(prof_path)
    from powergenome.params import build_resource_clusters
    cb = build_resource_clusters(group_path=rg_path, profile_path=prof_path)
    print(f"Loaded {len(cb.groups)} group(s)")
    for i, rg in enumerate(cb.groups):
        print(f"  [{i:2d}] tech={rg.group.get('technology')!r:<18} "
              f"existing={rg.group.get('existing')}  "
              f"extras={[k for k in rg.group if k not in ('technology','existing','tree','metadata','profiles','site_map')]}")
    print("\n--- find_groups replication ---")
    for tech in ["landbasedwind","utilitypv","offshorewind","hydro"]:
        for existing in [False, True]:
            hits = cb.find_groups(existing=existing, technology=tech)
            tag = "new-build" if not existing else "existing "
            mark = "✓" if hits else "✗ EMPTY"
            print(f"  {tag} {tech:<15} → {len(hits)}   {mark}")
except Exception as e:
    import traceback
    print("EXCEPTION during PG load:")
    traceback.print_exc()

# --- also check the ambient env vars the job sees ---
print(f"\n=== os.environ (before we overrode) ===")
# Re-read the process's ambient env — this is what PG would see if
# pg_to_switch doesn't explicitly set them
print(f"  RESOURCE_GROUPS         (pre-script): see below")
print(f"  RESOURCE_GROUP_PROFILES (pre-script): see below")
# (we already clobbered os.environ; the useful info is whether pg_to_switch
#  sets these in its own init — check by grepping that repo separately)