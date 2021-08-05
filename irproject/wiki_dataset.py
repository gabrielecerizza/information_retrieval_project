import bs4
import json
import os
import requests
import spacy
import wikipedia
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import List
from tqdm.auto import tqdm

spacy.prefer_gpu()


def get_related_pages(
    title_to_relate: str
):
    S = requests.Session()

    URL = f"https://en.wikipedia.org/api/" \
        + "rest_v1/page/related/{title_to_relate}"

    R = S.get(url=URL)
    DATA = R.json()

    related_pages_titles = [title_to_relate]

    for page in DATA["pages"]:
        related_pages_titles.append(page["title"])

    return related_pages_titles


def get_wikidata_info(title: str):
    S = requests.Session()

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "ppprop": "wikibase_item",
        "redirects": 1,
        "format": "json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    page_ids, wikidata_ids = [], []

    for _, v in DATA["query"]["pages"].items():
        page_ids.append(v["pageid"])
        wikidata_ids.append(v["pageprops"]["wikibase_item"])
    
    assert len(page_ids) == 1

    return page_ids[0], wikidata_ids[0]


def get_statements(wikidata_id: str):
    S = requests.Session()

    URL = "https://www.wikidata.org/w/api.php"

    PARAMS = {
        "action": "wbgetclaims",
        "entity": wikidata_id,
        "format": "json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    statements = defaultdict(lambda: [])
    related_entities = []

    for prop in DATA["claims"]:
        for el in DATA["claims"][prop]:
            if el["type"] == "statement":
                mainsnak = el["mainsnak"]
                if mainsnak["datatype"] == "wikibase-item":
                    entity = mainsnak["datavalue"]["value"]["id"]
                    statements[prop].append(entity)
                    related_entities.append(entity)

    return dict(statements), related_entities


def preprocess_paragraph(paragraph: bs4.element.Tag):
    for el in paragraph.find_all():
        if el.name == "sup":
            el.decompose()
            continue
        if (el.name != "a") or (not el.has_attr("title")):
            if el.name == None:
                continue
            el.unwrap()
    return paragraph


def process_paragraph(
    paragraph: bs4.element.Tag,
    title: str,
    page_id: str,
    page_wikidata_id: str,
    page_historical: int,
    page_categories: List[str],
    nlp: spacy.language.Language
):
    result = {
        "title": title,
        "historical": page_historical,
        "page_id": page_id,
        "wikidata_id": page_wikidata_id,
        "clean_content": paragraph.get_text(),
        "categories": page_categories,
        "entities": []
    }

    char_idx = 0
    word_idx = 0
    token_idx = 0

    for content in paragraph.contents:
        if content.name == "a":
            entity = dict()
            raw_text = content.get_text()
            text = content.get_text().strip()
            title = content.get("title")
            words_length = len(text.split(" "))
            tokens_length = len(nlp(text))
            entity_page_id, entity_wikidata_id = get_wikidata_info(
                title
            )
            entity["surface_form"] = raw_text
            entity["page_title"] = title
            entity["page_id"] = entity_page_id
            entity["wikidata_id"] = entity_wikidata_id
            entity["char_offset"] = char_idx
            entity["char_length"] = len(raw_text)
            entity["word_offset"] = word_idx
            entity["words_length"] = words_length
            entity["token_offset"] = token_idx
            entity["tokens_length"] = tokens_length
            entity["historical"] = is_historical(entity_wikidata_id)
            result["entities"].append(entity)
            char_idx += len(raw_text)
            word_idx += words_length
            token_idx += tokens_length
        else:
            raw_text = str(content)
            text = raw_text.strip()
            spl = text.split(" ")
            if spl[0] == "":
                words_length = 0
            else:
                words_length = len(spl)
            tokens_length = len(nlp(text))
            char_idx += len(raw_text)
            word_idx += words_length
            token_idx += tokens_length

    return result, token_idx


def process_page(
    page: wikipedia.WikipediaPage,
    title: str,
    page_id: str,
    page_wikidata_id: str,
    page_historical: int 
):
    nlp = spacy.load("en_core_web_sm")
    
    results = []
    total_tokens = 0

    html = page.html()
    soup = BeautifulSoup(html, "html.parser")

    paragraphs = soup.find_all("p")
    for paragraph in tqdm(
        paragraphs,
        desc="Processing paragraphs",
        leave=False
    ):
        if (paragraph.has_attr("class")) \
            and ("mw-empty-elt" in paragraph.get("class")):
            continue
        paragraph = preprocess_paragraph(paragraph)
        result, token_idx = process_paragraph(
            paragraph, title, page_id, 
            page_wikidata_id, page_historical, 
            page.categories, nlp
        )
        results.append(result)
        total_tokens += token_idx

    return results, total_tokens


def check_prop_entity(
    prop_id,
    entity_id,
    statements
):
    if prop_id not in statements:
        return False
    else:
        return entity_id in statements[prop_id]


def update_dataset(
    seed_title: str,
    reset: bool = False,
    filename: str = "wiki_dataset"
):
    base_out_path = "datasets/historical_events"
    os.makedirs(base_out_path, exist_ok=True)

    if os.path.isfile(f"{base_out_path}/{filename}.json") \
        and not reset:
        with open(f"{base_out_path}/{filename}.json") as f_in:
            dataset = json.load(f_in)
    else:
        dataset = {
            "page_ids": [],
            "page_titles": [],
            "token_num": 0,
            "total_pages": 0,
            "total_historical_pages": 0,
            "total_other_pages": 0,
            "paragraphs": []
        }

    titles = get_related_pages(seed_title)

    for title in tqdm(
        titles,
        desc="Processing titles",
        leave=False
    ):
        if title not in dataset["page_titles"]:
            print(title)
            page = wikipedia.page(title)
            assert not page.pageid in dataset["paged_ids"]

            dataset["pages_ids"].append(page.pageid)
            dataset["page_titles"].append(title)
            dataset["totale_pages"] += 1

            page_id, page_wikidata_id = get_wikidata_info(title)
            assert page_id == page.pageid

            page_historical = is_historical(page_wikidata_id)
            if page_historical:
                dataset["total_historical_pages"] += 1
            else:
                dataset["total_other_pages"] += 1

            paragraphs, total_tokens = process_page(
                page, page_id, page_wikidata_id, page_historical
            )
            dataset["paragraphs"].extend(paragraphs)
            dataset["token_num"] += total_tokens

    with open(
        f"{base_out_path}/{filename}.json", "w", 
        encoding="utf-8"
    ) as f_out:
        json.dump(
            dataset, f_out, 
            ensure_ascii=False, 
            indent=4
        )

    return dataset


def load_dataset(
    filename: str = "wiki_dataset"
):
    base_out_path = "datasets/historical_events"
    
    with open(f"{base_out_path}/{filename}.json") as f_in:
        return json.load(f_in)


def is_historical(wikidata_id: str):
    """True = 1; False = 0."""
    
    # Property map.
    p = {
        "instance of": "P31",
        "topic's main Wikimedia portal": "P1151",
        "topic's main category": "P910",
        "history of topic": "P2184",
        "conflict": "P607",
        "part of": "P361",
        "occupation": "P106",
        "noble title": "P97",
        "monogram": "P1543",
        "subclass of": "P279",
        "studied by": "P2579",
        "military branch": "P241",
        "position held": "P39",
        "member of political party": "P102"
    }

    # Entity map.
    e = {
        "historical period": "Q11514315",
        "Portal:World War II": "Q3247957",
        "Category:World War II": "Q6816704",
        "siege": "Q188055",
        "war": "Q198",
        "Napoleonic Wars": "Q78994",
        "politician": "Q82955",
        "revolution": "Q10931",
        "Portal:French Revolution": "Q3247542",
        "Category:French Revolution": "Q7216178",
        "battle": "Q178561",
        "combat": "Q650711",
        "historical event": "Q13418847",
        "history": "Q309",
        "military operation": "Q645883",
        "study of history": "Q1066186",
        "Category:Historical events": "Q32571532",
        "military officer": "Q189290",
        "riot": "Q124757",
        "protest": "Q273120",
        "political crisis": "Q3002772",
        "diplomat": "Q193391",
        "political party": "Q7278",
        "civil war": "Q8465",
        "ethnic conflict": "Q766875",
        "military unit": "Q176799",
        "revolutionary": "Q3242115",
        "military unit branch class": "Q62933934",
        "political organization": "Q7210356",
        "militant": "Q17010072",
        "demonstration": "Q175331",
        "rebel": "Q1125062",
        "classical antiquity": "Q486761",
        "ancient history": "Q41493",
        "historical country": "Q3024240",
        "world war": "Q103495",
        "Portal:World War I": "Q10651811",
        "Category:World War I": "Q6816935",
        "conflict": "Q180684",
        "military campaign": "Q831663"
    }

    statements, related_entities = get_statements(wikidata_id)

    if check_prop_entity(p["instance of"], e["historical period"], statements) \
        or check_prop_entity(p["topic's main Wikimedia portal"], e["Portal:World War II"], statements) \
        or check_prop_entity(p["topic's main Wikimedia portal"], e["Portal:World War I"], statements) \
        or check_prop_entity(p["topic's main category"], e["Category:World War II"], statements) \
        or check_prop_entity(p["topic's main category"], e["Category:World War I"], statements) \
        or p["history of topic"] in statements \
        or p["position held"] in statements \
        or p["member of political party"] in statements \
        or p["conflict"] in statements \
        or p["military branch"] in statements \
        or check_prop_entity(p["instance of"], e["military campaign"], statements) \
        or check_prop_entity(p["instance of"], e["world war"], statements) \
        or check_prop_entity(p["instance of"], e["historical country"], statements) \
        or check_prop_entity(p["instance of"], e["conflict"], statements) \
        or check_prop_entity(p["instance of"], e["rebel"], statements) \
        or check_prop_entity(p["subclass of"], e["rebel"], statements) \
        or check_prop_entity(p["instance of"], e["demonstration"], statements) \
        or check_prop_entity(p["instance of"], e["political organization"], statements) \
        or check_prop_entity(p["instance of"], e["military unit"], statements) \
        or check_prop_entity(p["instance of"], e["civil war"], statements) \
        or check_prop_entity(p["instance of"], e["ethnic conflict"], statements) \
        or check_prop_entity(p["instance of"], e["military unit branch class"], statements) \
        or check_prop_entity(p["part of"], e["classical antiquity"], statements) \
        or check_prop_entity(p["instance of"], e["siege"], statements) \
        or check_prop_entity(p["part of"], e["siege"], statements) \
        or check_prop_entity(p["subclass of"], e["siege"], statements) \
        or check_prop_entity(p["instance of"], e["war"], statements) \
        or check_prop_entity(p["part of"], e["war"], statements) \
        or check_prop_entity(p["subclass of"], e["war"], statements) \
        or check_prop_entity(p["part of"], e["ancient history"], statements) \
        or check_prop_entity(p["part of"], e["Napoleonic Wars"], statements) \
        or check_prop_entity(p["occupation"], e["politician"], statements) \
        or check_prop_entity(p["occupation"], e["diplomat"], statements) \
        or check_prop_entity(p["occupation"], e["militant"], statements) \
        or check_prop_entity(p["occupation"], e["revolutionary"], statements) \
        or p["noble title"] in statements \
        or p["monogram"] in statements \
        or check_prop_entity(p["instance of"], e["revolution"], statements) \
        or check_prop_entity(p["instance of"], e["political party"], statements) \
        or check_prop_entity(p["topic's main Wikimedia portal"], e["Portal:French Revolution"], statements) \
        or check_prop_entity(p["topic's main category"], e["Category:French Revolution"], statements) \
        or check_prop_entity(p["instance of"], e["battle"], statements) \
        or check_prop_entity(p["part of"], e["battle"], statements) \
        or check_prop_entity(p["subclass of"], e["battle"], statements) \
        or check_prop_entity(p["instance of"], e["combat"], statements) \
        or check_prop_entity(p["part of"], e["combat"], statements) \
        or check_prop_entity(p["subclass of"], e["combat"], statements) \
        or check_prop_entity(p["subclass of"], e["historical event"], statements) \
        or check_prop_entity(p["instance of"], e["historical event"], statements) \
        or check_prop_entity(p["subclass of"], e["military operation"], statements) \
        or check_prop_entity(p["instance of"], e["military operation"], statements) \
        or check_prop_entity(p["studied by"], e["study of history"], statements) \
        or check_prop_entity(p["topic's main category"], e["Category:Historical events"], statements) \
        or check_prop_entity(p["occupation"], e["military officer"], statements) \
        or check_prop_entity(p["instance of"], e["riot"], statements) \
        or check_prop_entity(p["instance of"], e["protest"], statements) \
        or check_prop_entity(p["instance of"], e["political crisis"], statements):
        return 1

    return 0





