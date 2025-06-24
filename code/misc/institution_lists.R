## convert openalex institutions list to csv
library(glue)
library(tidyverse)
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
filename <- glue("{root}/data/crosswalks/institutions.rds")
inst = readRDS(filename)

write_csv(filter(inst, is.na(country_code) & type == 'education'), glue("{root}/data/crosswalks/institutions_manual_entry.csv"))

# list of acronyms
acronyms <- inst %>% select(c(id, name, country_code, acronyms, type)) %>% unnest(cols = acronyms)
# write_csv(acronyms,glue("{root}/data/crosswalks/institutions_acronyms.csv"))

# list of alternative names
altnames <- inst %>% select(c(id, name, country_code, alternative_names, type)) %>% unnest(cols = alternative_names)
# write_csv(altnames,glue("{root}/data/crosswalks/institutions_altnames.csv"))

# names (no acronyms)
allnames <- bind_rows(list(select(inst, name, country_code, type),
                           select(altnames %>% mutate(name = alternative_names), name, country_code, type)))
