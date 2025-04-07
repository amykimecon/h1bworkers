# File Description: regressions
# Author: Amy Kim
# Date Created: Fri Apr  4 12:00:17 2025

# IMPORTING PACKAGES ----
library(tidyverse)
library(glue)
library(fixest)

# SETTING WORKING DIRECTORIES ----
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

df <- read_csv(glue("{root}/data/int/out_for_reg_apr4.csv"))
df[['post']] <- df[['q_rel']] >= 0

m1 <- feols(n_emp ~ i(q_rel, win_rate, 0) | main_rcid + q_rel + t^n_emp_max, data = df)

coefplot(m1, xlab = "win rate x quarter rel. to lottery")


m2 <- feols(new_positions ~ i(q_rel, win_rate, 0) | main_rcid + q_rel + t^n_emp_max, data = df)

coefplot(m2, xlab = "win rate x quarter rel. to lottery")



m1_ly2020 <- feols(n_emp ~ i(q_rel, win_rate, 0) | main_rcid + t^n_emp_max, data = df %>% filter(ly == 2021))
coefplot(m1_ly2020, xlab = 'win rate x quarter rel. to 2020 lottery')

m1_ly2021 <- feols(n_emp ~ i(q_rel, win_rate, 0) | main_rcid + t^n_emp_max, data = df %>% filter(ly == 2022))
coefplot(m1_ly2021, xlab = 'win rate x quarter rel. to 2021 lottery')

m1_ly2022 <- feols(n_emp ~ i(q_rel, win_rate, 0) | main_rcid + t^n_emp_max, data = df %>% filter(ly == 2023))
coefplot(m1_ly2022, xlab = 'win rate x quarter rel. to 2022 lottery')

m1_ly2023 <- feols(n_emp ~ i(q_rel, win_rate, 0) | main_rcid + t^n_emp_max, data = df %>% filter(ly == 2024))
coefplot(m1_ly2023, xlab = 'win rate x quarter rel. to 2023 lottery')

m1_post <- feols(n_emp ~ i(post, win_rate, FALSE) | main_rcid^ly + t^n_emp_max, data = df)

m2 <- feols(new_positions ~ i(q_rel, win_rate, -1) | main_rcid^ly + t^n_emp_max, data = df)
m2_post <- feols(new_positions ~ i(post, win_rate, FALSE) | main_rcid^ly, data = df %>% filter(ly))