import os, requests, datetime, warnings  # type: ignore[import-untyped]
import gseapy as gp
from bioservices.kegg import KEGG
from scripts.utils import make_valid_term
from scripts.output import read_gene_sets, save_gene_sets


### KEGG Annotations ###


common_organisms = {
    'human': 'hsa',
    'mouse': 'mmu',
    'zebrafish': 'dre',
    'fish': 'dre',
    'empedobacter brevis': 'ebv',
}

organisms_url = "http://rest.kegg.jp/list/organism"


def get_kegg_organism(organism):

    if organism in common_organisms.keys():
        return common_organisms[organism]

    response = requests.get(organisms_url)
    organisms = response.text.split("\n")

    for i in organisms:
        parts = i.split("\t")
        if len(parts) > 2 and organism in parts[2].lower():
            return parts[1]

    return None


def retrieve_all_kegg_pathways(organism: str, subset: int = 0) -> dict[str, list[str]]:

    # If organism is supported by MSigDB and includes KEGG, use it as it is much faster
    if organism == 'human':
        pathways = retrieve_all_msigdb_pathways(organism)
        return {name.replace('KEGG_', ''): pathway for name, pathway in pathways.items() if 'KEGG' in name}

    kegg_organism = get_kegg_organism(organism)
    if not kegg_organism:
        raise RuntimeError(f'Organism {organism} is not supported by KEGG annotations')

    k = KEGG()
    k.organism = kegg_organism
    pathway_list = k.pathwayIds

    if subset > 0:
        pathway_list = pathway_list[:subset]

    pathways = {}
    for kegg_id in pathway_list:
        try:
            pathway_info = k.parse(k.get(kegg_id))
            name = pathway_info['NAME'][0].split(' - ')[0]
            symbols = [gene.split(';')[0].strip() for gene in pathway_info['GENE'].values() if ';' in gene]
            if symbols and name:
                pathways[name] = symbols
        except:
            pass

    return pathways


### GO Annotations ###


def get_go_db(db):
    if db.lower().replace('_', ' ') in ['molecular function', 'mf']:
        return 'GO_Molecular_Function'
    elif db.lower().replace('_', ' ') in ['biological process', 'bp']:
        return 'GO_Biological_Process'
    elif db.lower().replace('_', ' ') in ['cellular component', 'cc']:
        return 'GO_Cellular_Component'
    raise ValueError(f'Invalid GO database {db}')


def get_library(db, organism):
    db = get_go_db(db)
    all_libraries = gp.get_library_name(organism=organism)

    curr_year = datetime.datetime.now().year
    for year in range(curr_year, 2017, -1):
        newest_db = [lib for lib in all_libraries if f'{db}_{year}' == lib]
        if newest_db:
            return newest_db[0]
    raise RuntimeError('No available GO databsae')
    

def get_go_pathways(db, organism):
    library = get_library(db, organism)
    pathways = gp.get_library(library, organism=organism)
    return pathways


def retrieve_all_go_pathways(organism: str, pathway_type: str | None = None) -> dict[str, list[str]]:
    
    pathway_types = [pathway_type] if pathway_type else ['bp', 'mf', 'cc']
    pathways = {}
    for pathway_type in pathway_types:
        pathways.update(get_go_pathways(pathway_type, organism))

    pathways = {key.split(' (GO')[0]: value for key, value in pathways.items()}
    return pathways


### MSigDB Annotations ###


msigdb_categories = {
    'Hs': ['h'] + [f'c{i}' for i in range(1, 9)],
    'Mm': ['mh', 'm1', 'm2', 'm3', 'm5', 'm8']
}


def get_msigdb():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return gp.Msigdb()


def get_msigdb_organism(organism: str) -> str | None:
    if organism.lower() in ['human', 'hs', 'homo sapiens']:
        return 'Hs'
    elif organism.lower() in ['mouse', 'mm', 'mus musculus']:
        return 'Mm'
    return None        


def get_category(category: str) -> str:
    category = category.lower()
    if '.' not in category:
        category = f'{category}.all'
    return category


def get_latest_ver(msig, organism):
    versions = msig.list_dbver()
    versions = versions[versions['Name'].str.contains(organism)]
    return versions['Name'].iloc[-1]


def get_msigdb_category(msig, category, organism):
    return msig.get_gmt(
        category=get_category(category),
        dbver=get_latest_ver(msig, organism)
    )


def retrieve_all_msigdb_pathways(organism: str, categories: list[str] | None = None) -> dict[str, list[str]]:
    msigdb_organism = get_msigdb_organism(organism)
    if not msigdb_organism:
        raise RuntimeError(f'Organism {organism} is not supported by MSigBD annotations - provide either `human` or `mouse`')
    
    msig = get_msigdb()

    pathways = {}
    categories = categories or msigdb_categories[msigdb_organism]
    for category in categories:
        pathways.update(get_msigdb_category(msig, category, msigdb_organism))
    return pathways


### Pathway Retrieval ###


def retrieve_pathway(id: str, organism: str) -> dict[str, list[str]]:
    # TODO: support pathway retrieval by ID
    # print('...')
    raise NotImplementedError('Pathway ID is not supported yet. Provide a valid pathway file instead.')


def intersect_genes(gene_set: list[str], all_genes: list[str], required_len: int = 1) -> list[str]:

    is_set = lambda gene_set: len(list(set(gene_set).intersection(set(all_genes)))) >= min(required_len, len(gene_set) // 2)
    intersect_set = lambda gene_set: sorted([g for g in set(gene_set) if g in all_genes])

    if is_set(gene_set):
        return intersect_set(gene_set)

    gene_set = [g.lower() for g in gene_set]
    if is_set(gene_set):
        return intersect_set(gene_set)

    gene_set = [g.upper() for g in gene_set]
    if is_set(gene_set):
        return intersect_set(gene_set)

    to_title = lambda word: word[0].upper() + word[1:].lower()
    gene_set = [to_title(g) for g in gene_set]
    if is_set(gene_set):
        return intersect_set(gene_set)

    return []


def get_gene_sets(pathway_database: list[str], custom_pathways: list[str], organism: str, all_genes: list[str], min_set_size: int, output: str) -> dict[str, list[str]]:
    gene_sets = {}

    for database in pathway_database:
        print(f'Retrieving pathways from {database}...')
        retrieval = globals()[f'retrieve_all_{database}_pathways']
        gene_sets.update(retrieval(organism))

    for pathway in custom_pathways: 
        # Read gene sets
        if os.path.exists(pathway):
            gene_sets.update(read_gene_sets(pathway))
        # Retrieve gene set from database by ID
        else:
            gene_sets.update(retrieve_pathway(pathway, organism))
    
    # Filter gene annotations based on size
    gene_sets = {make_valid_term(set_name): [g for g in intersect_genes(gene_set, all_genes) if g != []] for set_name, gene_set in gene_sets.items()}
    gene_sets = {set_name: gene_set for set_name, gene_set in gene_sets.items() if len(gene_set) >= min_set_size}

    # Save
    save_gene_sets(gene_sets, output)

    return gene_sets
