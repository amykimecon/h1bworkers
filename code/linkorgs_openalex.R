## fuzzymerge openalex institutions with revelio institutions using linkorgs
library(glue)
library(tidyverse)
library(LinkOrgs)
library(zoomerjoin)
library(tictoc)
LinkOrgs::BuildBackend(conda_env = "LinkOrgsEnv",  conda = "auto", tryMetal = TRUE)

root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"

# openalex institutions 
inst_clean <- read_csv(glue("{root}/data/int/allinstitutions_clean.csv"))

# linkedin institutions
univ_names <- read_csv(glue("{root}/data/int/rev_univ_names.csv"))
univ_names_nohs <- read_csv(glue("{root}/data/int/rev_univ_names_nohs.csv"))

# sample of linkedin institutions for testing
# univtest <- slice_sample(univ_names_nohs %>% filter(nchar(univ_raw_clean) >= 3), n = 1000)
# insttest <- inst_clean %>% filter(type == "education" & nchar(name_clean) >= 3)
univ_samp <- read_csv(glue("{root}/data/int/univ_samp.csv"))
inst_samp <- read_csv(glue("{root}/data/int/inst_samp.csv"))

# timing zoomerjoin fuzzy merge (jaccard)
tic()
zj_jaccard <- jaccard_inner_join(a = univ_samp %>% mutate(namejoin = university_raw),
                                 b = inst_samp %>% mutate(namejoin = name))
toc()


# trying default fuzzymerge
tic()
linkedOrgs_fuzzy2 <- LinkOrgs(x = univtest, y = insttest,
                             by.x = "univ_raw_clean", by.y = "name_clean",
                             algorithm = "fuzzy",
                             DistanceMeasure = "jaccard",
                             MaxDist = 0.7)
toc()

# zoomerjoin fuzzymerge
tic()
zoomerjoin_jaccard <- jaccard_inner_join(a = univtest %>% mutate(namejoin = univ_raw_clean),
                                         b = insttest %>% mutate(namejoin = name_clean))
toc()

linkedOrgs_ml2 <- LinkOrgs(x = univtest, y = insttest,
                             by.x = "univ_raw_clean", by.y = "name_clean",
                             algorithm = "ml",
                             DistanceMeasure = "jw",
                          AveMatchNumberPerAlias = 5,
                          conda_env = "LinkOrgsEnv",
                          conda_env_required = T)
linkedOrgs_ensemble2 <- LinkOrgs(x = univtest, y = insttest,
                                by.x = "univ_raw_clean", by.y = "name_clean",
                                AveMatchNumberPerAlias = 5,
                                algorithm = 'bipartite',
                                DistanceMeasure = 'ml',
                                conda_env = 'LinkOrgsEnv',
                                conda_env_required = T)