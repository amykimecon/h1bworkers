# File Description: Helper Functions for Merging Employer Data from Revelio and H1B Data
# Author: Amy Kim
# Date Created: Thu Feb 27
# import duckdb as ddb

# # Blocking on Company Name (Rare Tokens)
# def block_on_company_name(con, table1, table2, column, delimiter, rare_quantile = 0.99, limit = -1, show = True):
#     """
#     Merging two tables on company name (looking for at least one overlapping token)
#     """
#     # Tokenizing company name
#     tokenize(table1, f"{table1}_tokenized", column, delimiter, rare_quantile = rare_quantile, limit = limit, show = show)
#     tokenize(table2, f"{table2}_tokenized", column, delimiter, rare_quantile = rare_quantile, limit = limit, show = show)
    
#     # Blocking on rare tokens
#     block_on_rare_tokens(f"{table1}_tokenized", f"{table2}_tokenized", f"{table1}_blocked", f"{table2}_blocked", show = show)

# Tokenizing Company Name
def tokenize(con, table_in, table_out, column, delimiter, rare_quantile = 0.999, limit = -1, show = True):
    """
    Tokenizes a column in a table and creates a new table with the tokenized column
    """
    # indicates whether to limit the number of rows
    if (limit == -1):
        limstring = ""
    else:
        limstring = f"LIMIT {limit}"
    
    # create unique id and clean string
    #create_unique_id(con, table_in)
    #clean_company_string(con, table_in, column)

    # unnesting tokens and getting frequencies
    unnest_tokens(con, table_in, f"{column}_clean", delimiter)
    token_frequencies(con, table_in)

    # joining unnested and token freqs
    create_replace_table(con, f"""SELECT * FROM 
                         {table_in}_unnested JOIN {table_in}_token_freqs ON {table_in}_unnested.token = {table_in}_token_freqs.token""", f"{table_in}_unnested_with_freqs", show = False)
    
    # rare token freq cutoff
    token_freq_cutoff = con.sql(f"SELECT QUANTILE_CONT(freq, {rare_quantile} ORDER BY freq ASC) AS freq_num FROM {table_in}_token_freqs").df()['freq_num'][0]

    # crafting query (you were in the middle of rewriting this query to include a join on rare tokens only)
    query = f""" 
    SELECT * FROM (
        -- collapse tokens by unique_id into list with counts
        (SELECT unique_id,
            list_transform(
                list_zip(
                    array_agg(token ORDER BY token_id),
                    array_agg(freq ORDER BY token_id)
                ),
                x -> struct_pack(token := x[1], freq := x[2])
            ) as name_tokens_with_freq,
            array_agg(token ORDER BY token_id) as name_tokens
        FROM {table_in}_unnested_with_freqs
        GROUP BY unique_id) AS all_tokens
        LEFT JOIN (
        -- collapse rare tokens by unique_id into list (no counts)
        SELECT unique_id,
            array_agg(token ORDER BY token_id) as rare_name_tokens
        FROM {table_in}_unnested_with_freqs WHERE freq < {token_freq_cutoff}
        GROUP BY unique_id
        ) AS rare_tokens
        ON all_tokens.unique_id = rare_tokens.unique_id
        LEFT JOIN(
        -- get rarest token
        SELECT * FROM (SELECT unique_id, token as rarest_token, freq as rarest_freq,
            ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY freq ASC) AS rarity_rank,
        FROM {table_in}_unnested_with_freqs) WHERE rarity_rank = 1) AS rarest_token
        ON all_tokens.unique_id = rarest_token.unique_id
        LEFT JOIN(
        -- get second rarest token
        SELECT * FROM (SELECT unique_id, token as second_rarest_token, freq as second_rarest_freq,
            ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY freq ASC) AS rarity_rank,
        FROM {table_in}_unnested_with_freqs) WHERE rarity_rank = 2) AS second_rarest_token
        ON all_tokens.unique_id = second_rarest_token.unique_id
        RIGHT JOIN {table_in} raw
        ON all_tokens.unique_id = raw.unique_id
    ) {limstring}
    """
    create_replace_table(con, query, table_out, show = show)

def unnest_tokens(con, table_in, column, delimiter, show = False):
    """
    Unnests tokens in a column
    """
    query = f"""
    SELECT
        unique_id,
        unnest(regexp_split_to_array({column}, '{delimiter}')) AS token,
        generate_subscripts(regexp_split_to_array({column}, '{delimiter}'), 1) AS token_id
    FROM
        {table_in}
    """
    create_replace_table(con, query, f"{table_in}_unnested", show = show)

def token_frequencies(con, table_in, show = False):
    """
    Gets the frequency of tokens in a column
    """
    query = f"""
    SELECT
        token, COUNT(*)::float/(SELECT COUNT(*) FROM {table_in}_unnested) AS freq
    FROM {table_in}_unnested
    GROUP BY token
    ORDER BY freq DESC
    """
    create_replace_table(con, query, f"{table_in}_token_freqs", show = show)

## UTILITIES
def create_replace_table(con, query, table_out, show = True):
    """
    Creates a table with a query
    """
    query_out = f"""
    CREATE OR REPLACE TABLE {table_out} AS
    {query}
    """
    con.sql(query_out)
    if show:
        con.table(table_out).show(max_rows = 10)


def clean_company_string(column):
    """
    Cleans a company string
    """
    return(f"""
    REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower({column})), ',|\\.', '', 'g'), ' & ', ' and ', 'g'), '\\s*\\(.*\\)\\s*', ' ', 'g')
    """)

def get_matching_freqs(column, key, val, default = 0.0):
    """
    From matched table with l.column and r.column with structure [{key:key, val:val}], intersect keys and return zipped list of values with structure [{lval:lval, rval:rval}]
    """
    return(f"""
        -- First: intersecting keys 
        list_intersect(
           list_transform({column}_l, x -> x.{key}), list_transform({column}_r, x -> x.{key})
        -- Second: getting positions of intersecting keys in l and r and using to extract values
        ).list_transform(
            y -> struct_pack(
                lval := list_extract({column}_l, list_position(list_transform({column}_l, x -> x.{key}), y)).{val},
                rval := list_extract({column}_l, list_position(list_transform({column}_r, x -> x.{key}), y)).{val})
        ).list_concat([struct_pack(lval:={default},rval:={default})]) -- adding 0.0 for missing values
    """)

def dot_prod_matching_freqs(column, key, val):
    return(f"""
        {get_matching_freqs(column, key, val, default = 0.0)}.list_transform(x -> x.lval * x.rval).list_sum()
           """)

def mult_prod_matching_freqs(column, key, val):
    return(f"""
        {get_matching_freqs(column, key, val, default = 1)}.list_transform(x -> x.lval * x.rval).list_product()
           """)

def group_by_naics_code_foia(naics_digits = ""):
    return(f"""
    SELECT FEIN, 
        list_transform(
            list_zip(
                array_agg(NAICS{naics_digits}_CODE ORDER BY n_wins_naics_code DESC), 
                array_agg(n_wins_naics_code/n_wins_naics_tot ORDER BY n_wins_naics_code DESC)
            ), 
            x -> struct_pack(naics := x[1], share := x[2])
        ) as naics{naics_digits}_codes,
        array_agg(NAICS{naics_digits}_CODE ORDER BY n_wins_naics_code DESC).list_extract(1) AS top_naics{naics_digits}_code
    FROM (
        -- grouping by FEIN and NAICS code
        SELECT FEIN, NAICS{naics_digits}_CODE, MEAN(n_wins_naics_tot) AS n_wins_naics_tot, COUNT(*) AS n_wins_naics_code FROM employer_counts WHERE NOT NAICS{naics_digits}_CODE = 'NA' AND NOT NAICS_CODE = '999999'
        GROUP BY FEIN, NAICS{naics_digits}_CODE)
    GROUP BY FEIN
    """)

def group_by_naics_code_rev(naics_digits = ""):
    if naics_digits == "":
        substr_query = "naics_code"
    else:
        substr_query = f"SUBSTR(naics_code, 1, {naics_digits})"
    return(f"""SELECT rcid, 
        list_transform(
            list_zip(
                array_agg(naics{naics_digits}_code ORDER BY n_positions DESC),
                array_agg(n_positions/n_positions_us ORDER BY n_positions DESC)
            ),
            x -> struct_pack(naics := x[1], share := x[2])
        ) as naics{naics_digits}_codes,
        array_agg(naics{naics_digits}_code ORDER BY n_positions DESC).list_extract(1) AS top_naics{naics_digits}_code
    FROM (SELECT naics{naics_digits}_code, sum(n_positions) as n_positions, mean(n_positions_us) as n_positions_us, rcid FROM (SELECT n_positions, n_positions_us, rcid, {substr_query} AS naics{naics_digits}_code FROM rev_filt) WHERE naics{naics_digits}_code IS NOT NULL GROUP BY rcid, naics{naics_digits}_code)
    GROUP BY rcid""")