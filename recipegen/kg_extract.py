"""Evidence-bound, offline lexical extraction from original English recipe steps.

These are text mentions, not a complete ingredient inventory or a claim that a
video/image depicts the entity. No LLM, captions, POS tagger or duration summing.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Iterable


# canonical English | Chinese display label | explicit aliases (optional).
# This is a deliberately inspectable domain lexicon, not a noun-extraction model.
_INGREDIENT_ROWS = """
tomato|番茄
cherry tomato|圣女果|cherry tomatoes
plum tomato|李子番茄|roma tomato,roma tomatoes
potato|马铃薯
sweet potato|红薯
yam|山药类薯物
carrot|胡萝卜
onion|洋葱
red onion|红洋葱
white onion|白洋葱
yellow onion|黄洋葱
green onion|葱|scallion,spring onion
shallot|红葱头
garlic|大蒜|garlic clove
ginger|姜|ginger root
leek|韭葱
celery|芹菜|celery stalk
celeriac|根芹
broccoli|西兰花
cauliflower|菜花
cabbage|卷心菜
red cabbage|紫甘蓝
napa cabbage|大白菜|chinese cabbage
brussels sprout|抱子甘蓝
spinach|菠菜
kale|羽衣甘蓝
lettuce|生菜
romaine lettuce|罗马生菜|romaine
arugula|芝麻菜|rocket
watercress|西洋菜
chard|莙荙菜|swiss chard
bok choy|小白菜|pak choi,pak choy
beet|甜菜根|beetroot
radish|萝卜
daikon|白萝卜|daikon radish
turnip|芜菁
parsnip|欧防风根
cucumber|黄瓜
zucchini|西葫芦|courgette
eggplant|茄子|aubergine
pumpkin|南瓜
butternut squash|奶油南瓜
acorn squash|橡果南瓜
summer squash|夏南瓜
bell pepper|甜椒|sweet pepper
red bell pepper|红甜椒
green bell pepper|绿甜椒
yellow bell pepper|黄甜椒
chili pepper|辣椒|chilli pepper,chili,chilli
jalapeno|墨西哥辣椒|jalapeño
serrano pepper|塞拉诺辣椒
habanero|哈瓦那辣椒
mushroom|蘑菇
shiitake mushroom|香菇|shiitake
button mushroom|白蘑菇
portobello mushroom|波托贝洛蘑菇|portobello
oyster mushroom|平菇
corn|玉米|sweetcorn,sweet corn
pea|豌豆|green pea
snow pea|荷兰豆
sugar snap pea|甜豆|snap pea
green bean|四季豆|string bean
asparagus|芦笋
artichoke|朝鲜蓟
okra|秋葵
fennel|茴香球茎
bamboo shoot|竹笋
bean sprout|豆芽
avocado|牛油果
olive|橄榄
caper|酸豆|capers
apple|苹果
pear|梨
banana|香蕉
orange|橙子
lemon|柠檬
lime|青柠
grapefruit|葡萄柚
tangerine|橘子|mandarin orange
peach|桃
nectarine|油桃
plum|李子
apricot|杏
cherry|樱桃
strawberry|草莓
blueberry|蓝莓
raspberry|覆盆子
blackberry|黑莓
cranberry|蔓越莓
grape|葡萄
raisin|葡萄干
date|椰枣
fig|无花果
mango|芒果|mangoes
pineapple|菠萝
papaya|木瓜
kiwi|猕猴桃|kiwifruit
watermelon|西瓜
cantaloupe|哈密瓜
honeydew|蜜瓜|honeydew melon
coconut|椰子
pomegranate|石榴
passion fruit|百香果
dragon fruit|火龙果
rice|大米或米饭
brown rice|糙米
white rice|白米
basmati rice|印度香米
arborio rice|意大利烩饭米
wild rice|菰米
quinoa|藜麦
barley|大麦
oat|燕麦|rolled oat
bulgur|小麦碎粒
couscous|库斯库斯
millet|小米
buckwheat|荞麦
pasta|意大利面
spaghetti|意大利长面
penne|通心面
macaroni|空心面
lasagna|千层面|lasagne
noodle|面条
rice noodle|米粉
udon|乌冬面
bread|面包
white bread|白面包
whole wheat bread|全麦面包|wholemeal bread
breadcrumb|面包糠|bread crumb
tortilla|墨西哥薄饼
pita|皮塔饼|pita bread
flour|面粉
all purpose flour|通用面粉|all-purpose flour,plain flour
bread flour|高筋面粉
cake flour|低筋面粉
whole wheat flour|全麦粉|wholemeal flour
rice flour|米粉末
almond flour|杏仁粉
cornmeal|玉米粉
cornstarch|玉米淀粉|corn starch
tapioca starch|木薯淀粉|tapioca flour
potato starch|马铃薯淀粉
chicken|鸡肉
chicken breast|鸡胸肉
chicken thigh|鸡腿肉
chicken wing|鸡翅
turkey|火鸡肉
duck|鸭肉
beef|牛肉
ground beef|牛肉末|minced beef
beef steak|牛排|steak
beef brisket|牛胸肉|brisket
roast beef|烤牛肉
pork|猪肉
pork chop|猪排
pork belly|五花肉
ground pork|猪肉末|minced pork
ham|火腿
bacon|培根
sausage|香肠
lamb|羊肉
ground lamb|羊肉末|minced lamb
veal|小牛肉
fish|鱼|fishes
salmon|三文鱼
tuna|金枪鱼
cod|鳕鱼
tilapia|罗非鱼
trout|鳟鱼
sardine|沙丁鱼
anchovy|凤尾鱼
mackerel|鲭鱼
shrimp|虾|prawn
crab|螃蟹
lobster|龙虾
scallop|扇贝
clam|蛤蜊
mussel|贻贝
oyster|牡蛎
squid|鱿鱼|calamari
octopus|章鱼
egg|鸡蛋
egg white|蛋清
egg yolk|蛋黄
tofu|豆腐|bean curd
tempeh|天贝
chickpea|鹰嘴豆|garbanzo bean
lentil|扁豆
black bean|黑豆
kidney bean|芸豆
pinto bean|斑豆
white bean|白豆
edamame|毛豆
soybean|大豆|soya bean
split pea|豌豆瓣
milk|牛奶
whole milk|全脂牛奶
skim milk|脱脂牛奶|skimmed milk
evaporated milk|淡奶
condensed milk|炼乳|sweetened condensed milk
cream|奶油
heavy cream|厚奶油|double cream,whipping cream
sour cream|酸奶油
butter|黄油
unsalted butter|无盐黄油
salted butter|有盐黄油
ghee|酥油|clarified butter
yogurt|酸奶|yoghurt
greek yogurt|希腊酸奶|greek yoghurt
cheese|奶酪
cheddar cheese|切达奶酪|cheddar
mozzarella|马苏里拉奶酪|mozzarella cheese
parmesan|帕尔马奶酪|parmesan cheese
feta|菲达奶酪|feta cheese
cream cheese|奶油奶酪
ricotta|乳清奶酪|ricotta cheese
cottage cheese|茅屋奶酪
goat cheese|山羊奶酪
oil|食用油|cooking oil
olive oil|橄榄油|extra virgin olive oil,extra-virgin olive oil
vegetable oil|植物油
canola oil|菜籽油|rapeseed oil
sunflower oil|葵花籽油
sesame oil|芝麻油
coconut oil|椰子油
peanut oil|花生油|groundnut oil
salt|盐|table salt
sea salt|海盐
kosher salt|粗盐
pepper|胡椒或椒类
black pepper|黑胡椒|ground black pepper
white pepper|白胡椒
cayenne pepper|卡宴辣椒粉|cayenne
paprika|红椒粉
smoked paprika|烟熏红椒粉
chili powder|辣椒粉|chilli powder
red pepper flake|红辣椒碎|chili flake,chilli flake
cumin|孜然
coriander seed|芫荽籽
turmeric|姜黄
cinnamon|肉桂
nutmeg|肉豆蔻
clove|丁香
cardamom|小豆蔻
allspice|多香果
star anise|八角
bay leaf|月桂叶|bay leaves
basil|罗勒
oregano|牛至
thyme|百里香
rosemary|迷迭香
sage|鼠尾草
parsley|欧芹
cilantro|香菜|coriander leaves,fresh coriander
dill|莳萝
mint|薄荷
chive|细香葱
tarragon|龙蒿
marjoram|马郁兰
lemongrass|香茅
curry powder|咖喱粉
garam masala|印度混合香料
sugar|糖
white sugar|白糖|granulated sugar
brown sugar|红糖或黄砂糖
powdered sugar|糖粉|icing sugar,confectioners sugar,confectioner's sugar
honey|蜂蜜
maple syrup|枫糖浆
molasses|糖蜜
vanilla extract|香草精
vanilla|香草
cocoa powder|可可粉
chocolate|巧克力
dark chocolate|黑巧克力
white chocolate|白巧克力
chocolate chip|巧克力豆
baking powder|泡打粉
baking soda|小苏打|bicarbonate of soda,sodium bicarbonate
yeast|酵母
gelatin|明胶|gelatine
almond|杏仁
walnut|核桃
pecan|碧根果
hazelnut|榛子
peanut|花生|groundnut
cashew|腰果
pistachio|开心果
pine nut|松子
sesame seed|芝麻
poppy seed|罂粟籽
sunflower seed|葵花籽
pumpkin seed|南瓜籽
chia seed|奇亚籽
flaxseed|亚麻籽|flax seed
soy sauce|酱油|soya sauce
fish sauce|鱼露
oyster sauce|蚝油
worcestershire sauce|伍斯特酱
hot sauce|辣酱
tomato sauce|番茄酱汁
tomato paste|番茄膏
ketchup|番茄调味酱|tomato ketchup
mustard|芥末酱
dijon mustard|第戎芥末酱
mayonnaise|蛋黄酱|mayo
vinegar|醋
white vinegar|白醋
apple cider vinegar|苹果醋
balsamic vinegar|香醋
rice vinegar|米醋|rice wine vinegar
red wine vinegar|红酒醋
lemon juice|柠檬汁
lime juice|青柠汁
orange juice|橙汁
chicken broth|鸡汤|chicken stock
beef broth|牛肉汤|beef stock
vegetable broth|蔬菜汤|vegetable stock
broth|高汤|stock
water|水
wine|葡萄酒
white wine|白葡萄酒
red wine|红葡萄酒
beer|啤酒
rum|朗姆酒
brandy|白兰地
coconut milk|椰奶
almond milk|杏仁饮
oat milk|燕麦饮
peanut butter|花生酱
tahini|芝麻酱
miso|味噌
hoisin sauce|海鲜酱
sriracha|是拉差辣酱
pesto|青酱
salsa|莎莎酱
"""

_ACTION_ROWS = """
add|加入|adds,added,adding
bake|烘烤|bakes,baked,baking
blend|搅打|blends,blended,blending
boil|煮沸|boils,boiled,boiling
chop|切碎|chops,chopped,chopping
cook|烹煮|cooks,cooked,cooking
dice|切丁|dices,diced,dicing
fry|煎炸|fries,fried,frying
deep fry|油炸|deep-fry,deep fried,deep-fried,deep frying,deep-frying
stir fry|炒|stir-fry,stir fried,stir-fried,stir frying,stir-frying
grill|炙烤|grills,grilled,grilling
heat|加热|heats,heated,heating
preheat|预热|preheats,preheated,preheating
mix|混合|mixes,mixed,mixing
pour|倒入|pours,poured,pouring
roast|烤制|roasts,roasted,roasting
serve|装盘供食|serves,served,serving
slice|切片|slices,sliced,slicing
steam|蒸|steams,steamed,steaming
stir|搅拌|stirs,stirred,stirring
whisk|打匀|whisks,whisked,whisking
fold|翻拌|folds,folded,folding
knead|揉|kneads,kneaded,kneading
peel|去皮|peels,peeled,peeling
grate|刨丝|grates,grated,grating
mince|剁碎|minces,minced,mincing
crush|压碎|crushes,crushed,crushing
simmer|小火煮|simmers,simmered,simmering
saute|炒制|sauté,sauteed,sautéed,sauteing,sautéing
sear|煎封|sears,seared,searing
marinate|腌制|marinates,marinated,marinating
season|调味|seasons,seasoned,seasoning
drain|沥干|drains,drained,draining
rinse|冲洗|rinses,rinsed,rinsing
wash|清洗|washes,washed,washing
cool|冷却|cools,cooled,cooling
chill|冷藏|chills,chilled,chilling
freeze|冷冻|freezes,frozen,freezing
thaw|解冻|thaws,thawed,thawing
defrost|解冻|defrosts,defrosted,defrosting
remove|移出|removes,removed,removing
place|放置|places,placed,placing
transfer|转移|transfers,transferred,transferring
cover|盖上|covers,covered,covering
uncover|揭盖|uncovers,uncovered,uncovering
reduce|浓缩或降低|reduces,reduced,reducing
roll|擀或卷|rolls,rolled,rolling
shape|整形|shapes,shaped,shaping
press|压|presses,pressed,pressing
brush|刷涂|brushes,brushed,brushing
sprinkle|撒|sprinkles,sprinkled,sprinkling
garnish|点缀|garnishes,garnished,garnishing
toast|烘烤|toasts,toasted,toasting
broil|上火烤|broils,broiled,broiling
poach|水波煮|poaches,poached,poaching
braise|焖炖|braises,braised,braising
stew|炖|stews,stewed,stewing
beat|打发|beats,beaten,beating
whip|打发|whips,whipped,whipping
sift|过筛|sifts,sifted,sifting
strain|滤出|strains,strained,straining
caramelize|焦糖化|caramelizes,caramelized,caramelizing,caramelise,caramelised,caramelising
combine|混合|combines,combined,combining
cut|切|cuts,cutting
let rest|静置|rest,rests,rested,resting
"""

_TOOL_ROWS = """
knife|刀|chef's knife,paring knife
cutting board|砧板|chopping board
pan|锅
frying pan|平底煎锅|fry pan
skillet|煎锅
saucepan|长柄锅
pot|锅
stockpot|汤锅|stock pot
dutch oven|荷兰锅
wok|炒锅
oven|烤箱
microwave|微波炉|microwave oven
blender|搅拌机
immersion blender|手持搅拌器|stick blender
food processor|食物处理机
mixer|搅拌机
stand mixer|厨师机
hand mixer|手持打蛋器
bowl|碗
mixing bowl|搅拌碗
spatula|锅铲或刮刀
whisk|打蛋器|wire whisk,balloon whisk
spoon|勺子
wooden spoon|木勺
ladle|汤勺
tongs|夹子
fork|叉子
colander|滤水篮
strainer|滤网
sieve|筛子
peeler|削皮器
grater|刨丝器
rolling pin|擀面杖
baking sheet|烤盘|cookie sheet
baking tray|烤盘
baking pan|烤模
roasting pan|烤肉盘
sheet pan|烤盘
muffin tin|马芬模具|muffin pan
loaf pan|吐司模具
cake pan|蛋糕模具|cake tin
pie dish|派盘|pie pan
casserole dish|烤碗|baking dish
ramekin|小烤盅
grill|烤架|barbecue grill
steamer|蒸锅
pressure cooker|压力锅
slow cooker|慢炖锅
rice cooker|电饭锅
pepper mill|胡椒研磨器
garlic press|压蒜器
potato masher|土豆压泥器
egg beater|打蛋器|eggbeater
salad spinner|蔬菜甩干器
air fryer|空气炸锅
thermometer|温度计
kitchen scale|厨房秤
measuring cup|量杯
measuring spoon|量勺
can opener|开罐器
kitchen scissors|厨房剪刀|kitchen shears
piping bag|裱花袋
pastry brush|刷子
cooling rack|冷却架|wire rack
mortar|研钵
pestle|研杵
"""


@dataclass(frozen=True)
class LexiconEntry:
    normalized_name: str
    zh_name: str | None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Lexicon:
    ingredients: tuple[LexiconEntry, ...]
    actions: tuple[LexiconEntry, ...]
    tools: tuple[LexiconEntry, ...]


def _plural(term: str) -> str:
    """Derive variants only for a known lexical term; never stem arbitrary text."""
    prefix, sep, word = term.rpartition(" ")
    word = word or term
    irregular = {"tomato": "tomatoes", "potato": "potatoes", "leaf": "leaves", "knife": "knives", "loaf": "loaves", "fish": "fish", "beef": "beef", "rice": "rice", "flour": "flour", "water": "water", "pasta": "pasta", "spaghetti": "spaghetti", "broccoli": "broccoli", "asparagus": "asparagus", "celery": "celery", "garlic": "garlic", "couscous": "couscous", "tofu": "tofu", "tempeh": "tempeh", "milk": "milk", "butter": "butter"}
    if word in irregular:
        plural = irregular[word]
    elif len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        plural = word[:-1] + "ies"
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        plural = word + "es"
    else:
        plural = word + "s"
    return prefix + sep + plural


def _parse_rows(rows: str, *, pluralize: bool = False) -> tuple[LexiconEntry, ...]:
    entries = []
    for row in rows.strip().splitlines():
        parts = row.strip().split("|")
        canonical, chinese = parts[:2]
        aliases = [canonical] + (parts[2].split(",") if len(parts) == 3 else [])
        if pluralize:
            aliases += [_plural(alias) for alias in list(aliases)]
        entries.append(LexiconEntry(canonical, chinese, tuple(dict.fromkeys(aliases))))
    return tuple(entries)


DEFAULT_LEXICON = Lexicon(
    ingredients=_parse_rows(_INGREDIENT_ROWS, pluralize=True),
    actions=_parse_rows(_ACTION_ROWS),
    tools=_parse_rows(_TOOL_ROWS, pluralize=True),
)


def _surface_key(text: str) -> str:
    return re.sub(r"[\s\-‐‑–—]+", " ", text.lower()).strip()


def load_lexicon(path: str | Path, *, include_defaults: bool = True) -> Lexicon:
    """Load literal custom terms, extending or replacing the inspectable defaults.

    JSON keys: ingredients/actions/tools, each an array of objects containing
    normalized_name, optional zh_name, optional aliases (a string array).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {"ingredients", "actions", "tools"}:
        raise ValueError("词典顶层仅接受 ingredients、actions、tools 三类数组")
    output = {}
    for category in ("ingredients", "actions", "tools"):
        entries = {entry.normalized_name: entry for entry in getattr(DEFAULT_LEXICON, category)} if include_defaults else {}
        rows = raw.get(category, [])
        if not isinstance(rows, list) or len(rows) > 10_000:
            raise ValueError(f"{category} 必须是最多 10000 项的数组")
        for row in rows:
            if not isinstance(row, dict) or set(row) - {"normalized_name", "zh_name", "aliases"}:
                raise ValueError("词条仅接受 normalized_name、zh_name、aliases")
            canonical = row.get("normalized_name")
            aliases = row.get("aliases", [])
            chinese = row.get("zh_name")
            if not isinstance(canonical, str) or not canonical.strip() or len(canonical) > 200:
                raise ValueError("normalized_name 必须是 1~200 字符的字符串")
            if chinese is not None and (not isinstance(chinese, str) or len(chinese) > 200):
                raise ValueError("zh_name 必须为字符串或 null")
            if not isinstance(aliases, list) or len(aliases) > 100 or any(not isinstance(alias, str) or not alias.strip() or len(alias) > 200 for alias in aliases):
                raise ValueError("aliases 必须是最多 100 个非空字符串")
            canonical = canonical.lower().strip()
            previous = entries.get(canonical)
            surfaces = list(previous.aliases if previous else ()) + [canonical] + [alias.strip().lower() for alias in aliases]
            if category in {"ingredients", "tools"}:
                surfaces += [_plural(alias) for alias in list(surfaces)]
            entries[canonical] = LexiconEntry(canonical, chinese if chinese is not None else (previous.zh_name if previous else None), tuple(dict.fromkeys(surfaces)))
        output[category] = tuple(entries.values())
    return Lexicon(**output)


def _compile_entries(entries: tuple[LexiconEntry, ...]) -> tuple[re.Pattern[str] | None, dict[str, LexiconEntry]]:
    terms = {}
    for entry in entries:
        for surface in (entry.normalized_name, *entry.aliases):
            key = _surface_key(surface)
            previous = terms.get(key)
            if previous is not None and previous.normalized_name != entry.normalized_name:
                raise ValueError(f"词典别名冲突：{surface!r} → {previous.normalized_name!r} / {entry.normalized_name!r}")
            terms[key] = entry
    if not terms:
        return None, terms
    # Long phrases win at the same start position; a single non-overlapping
    # matcher prevents tomato/olive/oil from duplicating their longer concepts.
    ordered = sorted(terms, key=lambda value: (-len(value.split()), -len(value), value))
    patterns = [r"[\s\-‐‑–—]+".join(re.escape(word) for word in key.split(" ")) for key in ordered]
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(patterns) + r")(?![A-Za-z0-9_])", re.IGNORECASE), terms


_NUMBER = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|[¼½¾])"
_NUMBER_RANGE = _NUMBER + r"(?:\s*(?:-|–|—|to)\s*" + _NUMBER + r")?"
_DURATION = re.compile(r"(?<![A-Za-z0-9_])(?:" + _NUMBER_RANGE + r"\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)|half\s+(?:an?\s+)?hour|an?\s+(?:minute|hour)|overnight)(?![A-Za-z0-9_])", re.IGNORECASE)
_TEMPERATURE = re.compile(r"(?<![A-Za-z0-9_])[-+]?" + _NUMBER_RANGE + r"\s*(?:°\s*[CF]|degrees?\s*(?:Celsius|Fahrenheit|[CF])|Celsius|Fahrenheit|[CF])(?![A-Za-z0-9_])", re.IGNORECASE)
_AMBIGUOUS_TOOLS = {"whisk", "grill"}
_TOOL_CONTEXT = re.compile(r"(?:\b(?:a|an|the|with|using|in|into|on|onto|from|your)\s+(?:(?:wire|balloon|clean|large|small|hot)\s+)?)$", re.IGNORECASE)


class RuleExtractor:
    def __init__(self, lexicon: Lexicon | None = None):
        self.lexicon = lexicon or DEFAULT_LEXICON
        self.matchers = {kind: _compile_entries(entries) for kind, entries in [
            ("Ingredient", self.lexicon.ingredients), ("Tool", self.lexicon.tools), ("Action", self.lexicon.actions)
        ]}

    def extract_step(self, text: str, step_order: int, *, source: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            raise TypeError("步骤 text 必须为原始字符串")
        if isinstance(step_order, bool) or not isinstance(step_order, int) or step_order < 1:
            raise ValueError("step_order 必须为正整数")
        if source is not None and not isinstance(source, dict):
            raise TypeError("source 必须是来源元数据对象或 None")
        mentions = []
        noun_spans = []

        def emit(kind: str, match: re.Match[str], normalized: str, chinese: str | None = None):
            mention = {"kind": kind, "name": match.group(0), "normalized_name": normalized,
                       "step_order": step_order,
                       "evidence": {"text": text, "span_start": match.start(), "span_end": match.end()},
                       "method": "rule", "confidence": None}
            if chinese is not None:
                mention["zh_name"] = chinese
            if source is not None:
                mention["source"] = deepcopy(source)
            mentions.append(mention)

        # Prefer whole tool phrases over contained food nouns (rice cooker,
        # garlic press, stock pot), then ingredients over their verb modifiers.
        for kind in ("Tool", "Ingredient", "Action"):
            pattern, terms = self.matchers[kind]
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                entry = terms[_surface_key(match.group(0))]
                if kind == "Tool" and entry.normalized_name in _AMBIGUOUS_TOOLS and _surface_key(match.group(0)) in {entry.normalized_name, _plural(entry.normalized_name)}:
                    if not _TOOL_CONTEXT.search(text[max(0, match.start() - 50):match.start()]):
                        continue
                if kind in {"Ingredient", "Action"} and any(match.start() < end and match.end() > start for start, end in noun_spans):
                    continue  # rice cooker / baking powder are noun phrases.
                emit(kind, match, entry.normalized_name, entry.zh_name)
                if kind in {"Ingredient", "Tool"}:
                    noun_spans.append(match.span())
        for kind, pattern in (("Duration", _DURATION), ("Temperature", _TEMPERATURE)):
            for match in pattern.finditer(text):
                emit(kind, match, re.sub(r"\s+", " ", match.group(0).strip().lower()))
        return sorted(mentions, key=lambda item: (item["evidence"]["span_start"], item["kind"], item["evidence"]["span_end"]))

    def extract_steps(self, steps: Iterable[str | dict[str, Any]], *, source: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if isinstance(steps, (str, bytes, dict)):
            raise TypeError("steps 必须为步骤数组，不能是单个字符串或对象")
        result = []
        for index, step in enumerate(steps, 1):
            if isinstance(step, str):
                result.extend(self.extract_step(step, index, source=source))
            elif isinstance(step, dict) and isinstance(step.get("text"), str):
                result.extend(self.extract_step(step["text"], step.get("order", index), source=step.get("source", source)))
            else:
                raise TypeError("每一步必须是字符串或含原始 text 的对象")
        return result


@lru_cache(maxsize=4)
def _extractor(lexicon: Lexicon | None = None) -> RuleExtractor:
    return RuleExtractor(lexicon)


def extract_step(text: str, step_order: int, *, source: dict[str, Any] | None = None, lexicon: Lexicon | None = None) -> list[dict[str, Any]]:
    return _extractor(lexicon).extract_step(text, step_order, source=source)


def extract_steps(steps: Iterable[str | dict[str, Any]], *, source: dict[str, Any] | None = None, lexicon: Lexicon | None = None) -> list[dict[str, Any]]:
    return _extractor(lexicon).extract_steps(steps, source=source)
