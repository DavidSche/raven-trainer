import json

# 1. Add missing translation keys
with open("ui_src/src/i18n/locales/en.json", encoding="utf-8") as f:
    en = json.load(f)

en["ingestionProfiles"]["dataset"] = "Dataset"
en["ingestionProfiles"]["targetDataset"] = "Target Dataset"  
en["ingestionProfiles"]["newProfile"] = "New Ingestion Profile"
en["ingestionProfiles"]["editing"] = "Editing"
en["ingestionProfiles"]["conceptName"] = "IngestionProfile"
en["ingestionProfiles"]["noProfilesHint"] = "Create an Ingestion Profile to define how raw data is processed into asset datasets."
en["ingestionProfiles"]["autoDerive"] = "(will auto-derive from name)"

with open("ui_src/src/i18n/locales/en.json", "w", encoding="utf-8") as f:
    json.dump(en, f, indent=2, ensure_ascii=False)

with open("ui_src/src/i18n/locales/zh.json", encoding="utf-8") as f:
    zh = json.load(f)

zh["ingestionProfiles"]["dataset"] = "数据集"
zh["ingestionProfiles"]["targetDataset"] = "目标数据集"
zh["ingestionProfiles"]["newProfile"] = "新建接入配置"
zh["ingestionProfiles"]["editing"] = "编辑中"
zh["ingestionProfiles"]["conceptName"] = "接入配置"
zh["ingestionProfiles"]["noProfilesHint"] = "创建一个Ingestion Profile来定义如何处理原始数据。"
zh["ingestionProfiles"]["autoDerive"] = "（自动从名称推导）"

with open("ui_src/src/i18n/locales/zh.json", "w", encoding="utf-8") as f:
    json.dump(zh, f, indent=2, ensure_ascii=False)

print("Translation keys added")

# 2. Fix IngestionProfilesPage.tsx
with open("ui_src/src/pages/IngestionProfilesPage.tsx", encoding="utf-8") as f:
    content = f.read()

# L178: Dataset label in source node
content = content.replace(
    '<label className="text-[10px] text-gray-500 font-medium block mb-1">Dataset</label>',
    '<label className="text-[10px] text-gray-500 font-medium block mb-1">{t(\'ingestionProfiles.dataset\')}</label>'
)

# L582: Target Dataset label in output node
content = content.replace(
    '<label className="mb-1 block text-[10px] text-gray-500 font-medium">Target Dataset</label>',
    '<label className="mb-1 block text-[10px] text-gray-500 font-medium">{t(\'ingestionProfiles.targetDataset\')}</label>'
)

# L619: New Ingestion Profile string in template literal
content = content.replace(
    '"New Ingestion Profile"',
    "t('ingestionProfiles.newProfile')"
)

# L620: "ing" badge (editing indicator)
content = content.replace(
    '">ing</span>}',
    "\">{t('ingestionProfiles.editing')}</span>}"
)

# L583: "(will auto-derive from name)"
content = content.replace(
    '"(will auto-derive from name)"',
    "t('ingestionProfiles.autoDerive')"
)

# L805: IngestionProfile in flow diagram
content = content.replace(
    '<strong>IngestionProfile</strong>',
    '<strong>{t(\'ingestionProfiles.conceptName\')}</strong>'
)

# L816: Empty state description
content = content.replace(
    '<p className="text-xs text-gray-400 mb-6">Create an Ingestion Profile to define how raw data is processed into asset datasets.</p>',
    '<p className="text-xs text-gray-400 mb-6">{t(\'ingestionProfiles.noProfilesHint\')}</p>'
)

with open("ui_src/src/pages/IngestionProfilesPage.tsx", "w", encoding="utf-8") as f:
    f.write(content)

print("IngestionProfilesPage.tsx: fixed low-risk strings")
