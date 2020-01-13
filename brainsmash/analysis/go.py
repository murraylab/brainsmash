""" Functions to perform gene ontology (GO) enrichment analyses.

Map -> enrichments
(Surrogate) Maps -> enrichments

"""

from ..maps.core import Base
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.anno.gaf_reader import GafReader
from goatools.obo_parser import GODag
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import pkg_resources
import requests
import logging
import io
import gzip
import os

# TODO this whole module is a disaster

__all__ = ['init_go_enrichment_study', 'spatial_go_enrichment_analysis',
           'spatial_pearsonr', 'mk_surrogates', 'get_gene_expression_ahba',
           'spatial_go_enrichment_analysis2']


def rsrc(x):
    return pkg_resources.resource_filename('surrogates', x)


def _maybe_download(fname, url):
    if not os.path.exists(fname):
        print("downloading")
        r = requests.get(url)
        out_content = r.content

        if url.endswith('.gz'):
            compressed_file = io.BytesIO(out_content)
            decompressed_file = gzip.GzipFile(fileobj=compressed_file)
            out_content = decompressed_file.read()

        with open(fname, 'wb') as f:
            print("writing file")
            f.write(out_content)
    else:
        print("file exists")


def _get_go(url="http://purl.obolibrary.org/obo/go/go-basic.obo"):
    go_fname = rsrc("data/go/go-basic.obo")
    _maybe_download(go_fname, url)
    return go_fname


def _get_gaf(url="http://geneontology.org/gene-associations/goa_human.gaf.gz"):
    gaf_fname = rsrc('data/go/goa_human.gaf')
    _maybe_download(gaf_fname, url)
    return gaf_fname


def init_go_enrichment_study(
        background=None, stat_methods=("bonferroni",), alpha=0.05, **kwargs):
    """
    Initialize GO Enrichment Study object.

    Parameters
    ----------
    background : list
        uniprot-ids
    stat_methods : TODO
    alpha : float
        significance threshold

    Returns
    -------
    goatools.go_enrichment.GOEnrichmentStudy instance

    Notes
    -----
    See
    https://github.com/tanghaibao/goatools/blob/master/notebooks/goea_nbt3102.ipynb
    """

    go_dag = GODag(_get_go())
    gaf_reader = GafReader(_get_gaf())

    assocs = {}
    for a in gaf_reader.associations:
        assocs.setdefault(a.DB_ID, set()).add(a.GO_ID)

    if background is None:
        logging.warning(
            "No background specified, using full set of available IDs")
        background = list(assocs.keys())
    elif isinstance(background, (list, tuple, np.ndarray)):
        pass
    else:
        raise ValueError("Invalid ``background``")

    go_enrichment_study = GOEnrichmentStudy(
        background, assocs, go_dag, propagate_counts=True, alpha=alpha,
        methods=stat_methods, **kwargs)

    return go_enrichment_study


def spatial_go_enrichment_analysis(
        topography, gene_topographies, distances, go_enrichment_study,
        r_thr=0.2, p_thr=0.05, **surrogate_kwargs):
    """
    Find enriched GO terms for genes whose expression topography is similar to
    the target topography.

    Step 1) Calculate Pearson correlation coefficients between target topography
        and all gene expression topographies and select those genes for which
        the correlation exceeds `r_thr`.
    Step 2) Perform enrichment analysis on resulting gene list and select GO
        terms for which Bonferroni-corrected p-value does not exceed `p_thr`
    Step 3) Repeat steps 1 and 2 for each surrogate topography, resulting in a
        list of Bonferroni-corrected p-values per surrogate for each GO term
    Step 4) For each significant GO term from step 2, calculate the probability
        that a surrogate p-value is smaller than the actual p-value.

    Parameters
    ----------
    topography : np.ndarray (n_locations,)
        The topography to be compared to the gene topographies
    gene_topographies : pd.DataFrame (n_genes x n_locations)
        Gene expression table. Index must contain uniprot-ids
    distances : np.ndarray (n_locations,n_locations)
        distance matrix
    go_enrichment_study : goatools.go_enrichment.GOEnrichmentStudy instance
        Object used to perform enrichment analyses
    r_thr : float
        Gene lists for enrichment analysis are constructed such that all genes
        are included for which the correlation between the gene's topography
        and the target topography is at least `r_thr`
    p_thr : float
        After enrichment analysis, keep only those GO terms for which the
        Bonferroni-corrected p-value is smaller than `p_thr`
    surrogate_kwargs
        Keyword arguments passed to ``construct_nulls``

    Returns
    -------
    pd.Series (n_significant_GO_terms,)
        Entries are the probabilities that the p-value for a surrogate's
        enrichment analysis is smaller than the actual topography's p-value
    """

    assoc_gene_ids = set(go_enrichment_study.assoc.keys())
    gene_topography_ids = set(gene_topographies.index.values)
    frac_found_ids = 1. * len(gene_topography_ids.intersection(
        assoc_gene_ids)) / len(gene_topography_ids)

    if frac_found_ids < 1:
        logging.warning(
            'Only {:.0f}% of gene-ids are present in the association table. '
            'Check that the index of gene_topographies contains uniprot-ids '
            'and/or consider setting a background gene list'.format(
                100 * frac_found_ids))

    gn = Base(topography, distances, **surrogate_kwargs)
    surrogate_topos = gn(n=100)

    r_topo = 1 - cdist(gene_topographies.values, topography.reshape(
        1, -1), metric='correlation')[:, 0]
    r_surr = 1 - cdist(
        gene_topographies.values, surrogate_topos, metric='correlation')

    # for the moment only consider absolute correlations
    r_topo = np.abs(r_topo)
    r_surr = np.abs(r_surr)

    topo_gene_list = gene_topographies.index.values[r_topo > r_thr]
    surr_gene_list = [gene_topographies.index.values[r_surr[:, surri] > r_thr]
                      for surri in range(r_surr.shape[1])]
    _n_surrogates = len(surr_gene_list)
    surr_gene_list = [sgl for sgl in surr_gene_list if len(sgl) > 0]

    if len(topo_gene_list) == 0:
        raise ValueError(
            'r_thr is too small; no strongly correlating topographies found.')
    if len(surr_gene_list) == 0:
        raise ValueError(
            'r_thr is too small; no surrogate topographies strongly '
            'correlate with one or more genes')
    elif len(surr_gene_list) < _n_surrogates:
        logging.warning(
            'Only {} of {} surrogates strongly correlate with at least 1 '
            'gene'.format(len(surr_gene_list), _n_surrogates))

    topo_ea_res = go_enrichment_study.run_study(topo_gene_list, log=None)
    surr_ea_res = [go_enrichment_study.run_study(
        surr_li, log=None) for surr_li in surr_gene_list]

    topo_p = pd.Series({g.GO: g.p_bonferroni for g in topo_ea_res}).sort_index()
    surr_p = pd.concat([
        pd.Series({g.GO: g.p_bonferroni for g in _surr_ea_res}).sort_index()
        for _surr_ea_res in surr_ea_res
    ], axis=1).sort_index()

    mask = topo_p < p_thr

    return (topo_p[mask] > surr_p[mask].T).mean(0)


def spatial_pearsonr(x, y, method='surrogates', x_surrogates=None,
                     y_surrogates=None, **surrogate_kwargs):
    """Compute Pearson correlation & spatial autocorrelation-corrected p-value.

    Parameters
    ----------
    x : np.ndarray (n_samples,)
    y : np.ndarray (n_samples,)
    method : str
        - 'naive': returns usual Pearson correlation coefficient p-value under
            the assumption of independent samples
        - 'surrogates': uses the method by Viladomat et al. to construct
            surrogates for x and y, then the histogram of absolute Pearson
            correlation coefficients between each x- and y-surrogate is used to
            construct a null distribution against which the absolute Pearson
            correlation of x and y is compared
        - 'clifford': uses the method by Clifford to calculate a corrected
            p-value (TODO: not implemented yet)
    x_surrogates : np.ndarray (n_x_surrogates, n_samples)
        ignored if method != 'surrogates'
    y_surrogates : np.ndarray (n_y_surrogates, n_samples)
        ignored if method != 'surrogates'
    **surrogate_kwargs
        passed to construct_nulls for case `method` == 'surrogates' when either
        x_surrogates or y_surrogates is None

    Returns
    -------
    r : float
        Pearson correlation coefficient
    p : float
        p-value computed according to `method` argument

    """

    r, p = pearsonr(x, y)

    # calculate corrected p-value
    if method == 'naive':
        return r, p

    elif method == 'surrogates':
        if x_surrogates is None:
            gn = Base(x, **surrogate_kwargs)
            x_surrogates = gn.generate(n=100)
        if y_surrogates is None:
            y_surrogates = Base(y, **surrogate_kwargs)
        null_rs = 1 - cdist(
            x_surrogates, y_surrogates, metric='correlation').ravel()
        p_corrected = np.mean(np.abs(null_rs) > np.abs(r))
        return r, p_corrected

    elif method == 'clifford':
        raise NotImplementedError()

    else:
        raise ValueError('Invalid method: {}'.format(method))


def _goenrichrec2series(rec):
    """Convert an enrichment record returned by a
    goatools.go_enrichment.GOEnrichmentStudy object into a pd.Series object.

    Parameters
    ----------
    rec : goatools.go_enrichment.GOEnrichmentStudy instance

    Returns
    -------
    pd.Series (n_go_features,)

    """

    s = pd.Series(dict(
        name=rec.name,
        NS=rec.NS,
        p_uncorrected=rec.p_uncorrected,
        p_bonferroni=rec.p_bonferroni,
        study_count=rec.study_count,
        study_n=rec.study_n,
        pop_count=rec.pop_count,
        pop_n=rec.pop_n,
    ))
    s.name = rec.GO
    return s


def mk_surrogates(topography, gene_topographies, distances, n_surrogates=100,
                  n_bins=15, deltas=tuple(np.arange(0.1, .901, .1)),
                  kernel='exp'):
    """Generates surrogates for target topography and gene topographies.

    Parameters
    ----------
    topography : np.ndarray (n_locations,)
        The target topography
    gene_topographies : pd.DataFrame (n_genes x n_locations)
        Table of gene topographies; index must contain uniprot-ids
    distances
    n_surrogates
    n_bins
    deltas
    kernel

    Returns
    -------
    topo_surrogates : np.ndarray (n_surrogates,n_locations)
    gene_topo_surrogates : np.ndarray (n_genes x n_surrogates,n_locations)
    """

    gn = Base(topography, distances, deltas=deltas, nbins=n_bins, kernel=kernel)
    topo_surrogates = gn(n=n_surrogates)

    gene_topo_surrogates = list()
    for _, g in gene_topographies.iterrows():
        gn = Base(
            g.values, distances, deltas=deltas, nbins=n_bins, kernel=kernel)
        nulls = gn(n=n_surrogates)
        gene_topo_surrogates.append(nulls)

    return topo_surrogates, np.vstack(gene_topo_surrogates)


def spatial_go_enrichment_analysis2(
        topography, gene_topographies, p_method='surrogates',
        topo_surrogates=None, gene_topo_surrogates=None,
        go_enrichment_study=None, topo_p_thr=0.05, ea_p_thr=0.05):
    """Find GO terms that are overrepresented in genes whose expression profile
    is significantly similar to a target topography.

    Step 1) Find genes that are significantly correlated with the target.
        topography by calculating spatial autocorrelation-corrected p-values
    Step 2) Filter p-values and keep those genes with p < `topo_p_thr`.
    Step 3) Perform enrichment analysis on resulting gene list, declaring GO
        terms significant if the Bonferroni-corrected p-value < `ea_p_thr`.

    Parameters
    ----------
    topography : np.ndarray (n_locations,)
        The target topography
    gene_topographies : pd.DataFrame (n_genes x n_locations)
        Table with gene expression topographies; index must contain uniprot-ids
    p_method : str
        'surrogates' or 'clifford'; forwarded to spatial_pearsonr
    topo_surrogates : np.ndarray (n_surrogates x n_locations)
        Pre-computed surrogate topographies; ignored if p_method != 'surrogates'
    gene_topo_surrogates : np.ndarray (n_genes x n_surrogates x n_locations)
        Pre-computed surrogate topographies for each gene; ignored if
        p_method != 'surrogates'
    go_enrichment_study : goatools.go_enrichment.GOEnrichmentStudy instance
        Used to perform enrichment analyses
    topo_p_thr : float
        Gene expression profiles are considered "significantly similar" to the
        target profile if the spatial autocorrelation-corrected p-value is below
        this threshold
    ea_p_thr : float
        GO terms are considered "significantly enriched" if the  Bonferroni-
        corrected p-value from the enrichment study is below this threshold

    Returns
    -------
    enriched_go_terms : pd.DataFrame (n_enriched_go_terms, n_go_features)
        Each row is an enriched GO term, and columns contain the properties
        (name, NS, p_uncorrected, p_bonferroni, study_count, study_n, pop_count,
        pop_n)

    """

    if p_method == 'surrogates':

        if len(gene_topographies) != len(gene_topo_surrogates):
            raise ValueError(
                'First dimension of gene_topographies and gene_topo_surrogates '
                'must have same length')

        rs, ps = zip(*[
            spatial_pearsonr(topography, g.values, method=p_method,
                             x_surrogates=topo_surrogates,
                             y_surrogates=gene_topo_surrogates[gi])
            for gi, (_, g) in enumerate(gene_topographies.iterrows())
                     ])
    
    elif p_method == 'clifford':
        rs, ps = zip(*[
            spatial_pearsonr(topography, g.values, method=p_method)
            for gi, (_, g) in enumerate(gene_topographies.iterrows())
                     ])
    else:
        raise ValueError('Invalid p_method: {}'.format(p_method))

    rs, ps = np.asarray(rs), np.asarray(ps)

    gene_list = gene_topographies.index.values[ps < topo_p_thr]

    gea_res = go_enrichment_study.run_study(gene_list)
    gea_res = [g for g in gea_res if g.p_bonferroni < ea_p_thr]
    enriched_go_terms = pd.concat(
        [_goenrichrec2series(g) for g in gea_res], axis=1).T
    enriched_go_terms.index.names = ['GO']

    return enriched_go_terms


def _add_uniprot_ids(gene_expression_table, multiple_ids=False):
    """
    TODO

    Parameters
    ----------
    gene_expression_table : TODO
    multiple_ids : bool
        TODO

    Returns
    -------
    TODO

    """

    # Load and clean data
    # -------------------

    tab = pd.read_csv(rsrc('data/go/genes/uniprot_30may2018.tsv'), sep='\t')
    tab['entrez_id'] = tab[  # ???
        'yourlist:M2018053083C3DD8CE55183C76102DC5D3A26728B1D278DS']
    del tab['yourlist:M2018053083C3DD8CE55183C76102DC5D3A26728B1D278DS']
    tab['Cross-reference (GeneID)'] = tab['Cross-reference (GeneID)']

    # Consistency check: assert that entrez_ids entered into UniProt
    # conversion tool match returned entrez_ids
    # --------------------------------------------------------------

    bad_rows = []
    for rid, r in tab.iterrows():
        given_ids = set(
            [int(x) for x in r['entrez_id'].split(',')])  # entered ids
        returned_ids = set(   # returned ids
            [int(x) for x in r['Cross-reference (GeneID)'].split(';')
             if x != ''])
        if len(given_ids.intersection(returned_ids)) == 0:
            bad_rows.append(rid)
    assert (len(bad_rows) == 0)

    # Create mapping from entrez_id to uniprot_id
    # -------------------------------------------

    entrez2uniprot = {}
    for rid, r in tab.iterrows():
        given_ids = set(
            [int(x) for x in r['entrez_id'].split(',')])
        returned_ids = set(
            [int(x) for x in r['Cross-reference (GeneID)'].split(';')
             if x != ''])
        all_entrez_ids = given_ids.intersection(returned_ids)
        for entrez_id in all_entrez_ids:
            entrez2uniprot.setdefault(entrez_id, set()).add(r['Entry'])

    # Check whether mapping is unique
    # -------------------------------

    multiple_targets = []
    for k, v in entrez2uniprot.items():
        if len(v) > 1:
            multiple_targets.append((k, v))
    logging.warning(
        'Found multiple mappings from entrez_id to uniprotid for {} '
        'entrez_ids'.format(len(multiple_targets)))

    # Apply mapping
    # -------------

    available_entrez_ids = gene_expression_table.index.get_level_values(
        'entrez_id').astype(np.int).values
    if multiple_ids:
        corresponding_uniprot_ids = [
            tuple(entrez2uniprot.get(entrez_id, set()))
            for entrez_id in available_entrez_ids]
    else:
        pick_first = lambda x: x[0] if len(x) > 0 else 'NA'
        corresponding_uniprot_ids = [pick_first(
            tuple(entrez2uniprot.get(entrez_id, set())))
            for entrez_id in available_entrez_ids]

    # Check for missing mappings
    # --------------------------

    n_no_mapping_found = len(
        [x for x in corresponding_uniprot_ids if len(x) == 0])
    logging.warning(
        'no mapping from entrez_id to uniprot_id found for {} '
        'entrez_ids'.format(n_no_mapping_found))

    # Add mappings to gene_expression_table
    # -------------------------------------

    gene_expression_table['uniprot_id'] = corresponding_uniprot_ids
    gene_expression_table.set_index('uniprot_id', append=True, inplace=True)

    return gene_expression_table


def get_gene_expression_ahba(
        gene_ids=('uniprot_id',), pick_one=True):

    gene_expression = pd.read_csv(
        rsrc('data/go/genes/gene_expression.csv'), index_col=0)
    gene_expression.columns = gene_expression.columns.astype(np.int)
    gene_expression.index.names = ['gene_symbol']

    if gene_ids != ('gene_symbol',):

        # Add entrez id
        # -------------

        aba_gene_set = pd.read_csv(
            rsrc('data/go/genes/gene_symbol_to_entrez_id.csv'))
        assert len(np.unique(aba_gene_set['gene_symbol'])) == len(aba_gene_set)

        # Check if genes in gene_expression exist in gene symbols map
        not_in_index = list(filter(
            lambda x: x not in aba_gene_set['gene_symbol'].values,
            gene_expression.index.values))
        assert(len(not_in_index) == 0)

        def pick_one_entrez_id(x):
            try:
                x = x.split('/')[0]
            except IndexError:
                return x
            else:
                return x

        if pick_one:
            aba_gene_set['entrez_id'] = aba_gene_set['entrez_id'].apply(
                pick_one_entrez_id)

        aba_gene_set.set_index('gene_symbol', inplace=True)

        gene_expression = pd.concat(
            [gene_expression, aba_gene_set], axis=1, sort=True).set_index(
            'entrez_id', append=True).dropna(axis=0, how='all')
        gene_expression.index.names = ['gene_symbol', 'entrez_id']

        # Add ensembl id
        # --------------

        def drop_entrezgene_acc(x):
            try:
                return str(x).split(':')[1]
            except IndexError:
                return np.nan

        gconvert_result_fname = rsrc(
            'data/go/genes/gprofiler_results_9feb2018.xlsx')
        gconvert_result = pd.read_excel(
            gconvert_result_fname, header=None,
            names=['qid', 'entrez_id', 'rid', 'ensembl_id', 'gene_symbol',
                   'desc', 'type'])

        gconvert_result['entrez_id'] = gconvert_result['entrez_id'].apply(
            drop_entrezgene_acc)
        gconvert_result.set_index('entrez_id', inplace=True)

        # in case g:Convert returned several ensemble_ids, select the first one
        if pick_one:
            pick_one_of_results = gconvert_result['rid'].apply(
                lambda x: np.allclose(float(x) % 1, .1))
            gconvert_result = gconvert_result.where(
                pick_one_of_results).dropna(axis=0, how='all')
            assert gconvert_result.index.is_unique

        gene_expression.reset_index('gene_symbol', inplace=True)
        gene_expression = pd.concat(
            [gene_expression, gconvert_result['ensembl_id']], axis=1, sort=True)
        gene_expression.index.names = ['entrez_id']
        gene_expression = gene_expression.set_index(
            ['gene_symbol', 'ensembl_id'], append=True).reorder_levels(
            ['gene_symbol', 'entrez_id', 'ensembl_id'])

        # Add uniprot_id
        # --------------
        _add_uniprot_ids(gene_expression)

    drop_ids = [i for i in gene_expression.index.names if i not in gene_ids]
    gene_expression = gene_expression.reset_index(drop_ids, drop=True)

    # do this at end, otherwise it gets overwritten when adding alternative ids
    gene_expression.columns.names = ['glasser_id']  # ???

    return gene_expression
