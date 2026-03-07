const fs = require("fs");
const path = require("path");
const REPO_ROOT = path.resolve(__dirname, "..");
const TM_ROOT = path.join(REPO_ROOT, "terraforming-mars");
const OUTPUT_PATH = path.join(REPO_ROOT, "card_metadata.json");
const EXPORT_SCRIPT = path.join(TM_ROOT, "src", "server", "tools", "export_card_metadata.ts");
const TS_CONFIG = path.join(TM_ROOT, "tsconfig.json");

const CARD_SECTIONS = [
  "projectCards",
  "corporationCards",
  "preludeCards",
  "ceoCards",
  "standardProjects",
  "standardActions",
];

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function ensureFileExists(filePath, label) {
  if (!fs.existsSync(filePath)) {
    fail(`Missing ${label}: ${filePath}`);
  }
}

function runBaseExport() {
  const previousCwd = process.cwd();
  try {
    process.chdir(REPO_ROOT);
    require(EXPORT_SCRIPT);
  } catch (error) {
    fail(`Base card metadata export failed.\n\n${error && error.stack ? error.stack : error}`);
  } finally {
    process.chdir(previousCwd);
  }
}

function registerTsNode() {
  process.env.TS_NODE_PROJECT = TS_CONFIG;
  require(path.join(TM_ROOT, "node_modules", "ts-node", "register", "transpile-only"));
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function collectRequirementMap() {
  const {ALL_MODULE_MANIFESTS} = require(path.join(TM_ROOT, "src", "server", "cards", "AllManifests"));
  const {CardManifest: CardManifestUtil} = require(path.join(TM_ROOT, "src", "server", "cards", "ModuleManifest"));

  const requirementMap = {};
  for (const manifest of ALL_MODULE_MANIFESTS) {
    for (const sectionName of CARD_SECTIONS) {
      const section = manifest[sectionName];
      if (!section) {
        continue;
      }
      for (const [, spec] of CardManifestUtil.entries(section)) {
        try {
          if (spec.instantiate === false) {
            continue;
          }
          const instance = new spec.Factory();
          const name = String(instance.name || "").trim();
          if (!name) {
            continue;
          }
          const requirements = Array.isArray(instance.requirements) ? cloneJson(instance.requirements) : [];
          requirementMap[name] = requirements;
        } catch (error) {
          process.stderr.write(`Skip requirement export for card: ${error}\n`);
        }
      }
    }
  }
  return requirementMap;
}

function augmentOutput(requirementMap) {
  const payload = JSON.parse(fs.readFileSync(OUTPUT_PATH, "utf8"));
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    fail(`Unexpected metadata shape in ${OUTPUT_PATH}`);
  }

  for (const [name, meta] of Object.entries(payload)) {
    if (!meta || typeof meta !== "object" || Array.isArray(meta)) {
      continue;
    }
    meta.requirements = Array.isArray(requirementMap[name]) ? requirementMap[name] : [];
  }

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(payload, null, 2), {encoding: "utf8"});
  process.stdout.write(`Wrote ${Object.keys(payload).length} cards with requirements to ${OUTPUT_PATH}\n`);
}

function main() {
  ensureFileExists(TM_ROOT, "Terraforming Mars checkout");
  ensureFileExists(path.join(TM_ROOT, "node_modules", "ts-node", "register", "transpile-only.js"), "ts-node register");
  ensureFileExists(EXPORT_SCRIPT, "base export script");
  ensureFileExists(TS_CONFIG, "Terraforming Mars tsconfig");

  registerTsNode();
  runBaseExport();
  const requirementMap = collectRequirementMap();
  augmentOutput(requirementMap);
}

main();
