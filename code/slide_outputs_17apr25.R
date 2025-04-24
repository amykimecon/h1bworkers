### 4/17 PRESENTATION SLIDE GRAPHS ####
library(tidyverse)
library(glue)
root <- "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code <- "/Users/amykim/Documents/GitHub/h1bworkers/code"

## country codes cw
iso_cw <- read_csv(glue("{root}/data/crosswalks/iso_country_codes.csv"))

## foia data
all_data_list <- list()
for (yr in 2021:2023){
  all_data_list[[yr - 2020]] <- read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY{yr}.csv"))
}
all_data_list[[4]] <- bind_rows(list(read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_single_reg.csv")),
                                     read_csv(glue("{root}/data/raw/foia_bloomberg/TRK_13139_FY2024_multi_reg.csv"))))
raw <- bind_rows(all_data_list)
raw$state_name <- c(state.name, "Washington, D.C.")[match(raw$state, c(state.abb, "DC"))]
raw$worksite_state_name <- state.name[match(raw$WORKSITE_STATE, state.abb)]

raw_countries <- data.frame(foia_country = unique(c(raw$country_of_birth, raw$country_of_nationality))) %>%
  left_join(iso_cw, by = c("foia_country" = "alpha-3")) %>%
  mutate(name_clean = str_remove(name, ",.*$"))

write_csv(raw_countries, glue("{root}/data/crosswalks/foia_countries.csv"))

## h1b applications by year
h1b_by_yr <- data.frame(yrs = 2005:2024, apps = c(85, 85, 123.5, 163, 85, 85, 85, 85, 124, 172.5,
                                                  233, 236, 199, 190, 201, 274, 308.5, 484, 781, 480))
h1b_by_yr['lot'] = ifelse(h1b_by_yr['apps']==85,"No Lottery","Lottery")

plot_out <- ggplot(h1b_by_yr, aes(x=yrs, y=apps, fill = lot)) + 
  geom_bar(stat = 'identity', alpha = 0.9) +
  labs(y = "H-1B Lottery Applications (Thousands)", fill = "", x = "Lottery Year") +
  scale_fill_manual(values = c("#9966b3", "grey")) +
  geom_hline(yintercept = 85, color = "#4f2f60") +
  geom_text(inherit.aes=FALSE,data = data.frame(x=c(2010.5),y=c(120),label=c("Cap: 85,000")),
            aes(label = label, x=x, y=y), size = 3.7, color = "#4f2f60") +
  theme_minimal()
  
## stats on prev visa class




plot_bar_graph <- function(df_in, filename, varname, ctrlname, ylab, tilt = FALSE){
  if (tilt){
    angle = 45
  }
  else{
    angle = 0
  }
  plot_out <- ggplot(df_in %>% mutate(ylb = estimate - 1.96*std.error, 
                                      yub = estimate + 1.96*std.error,
                                      pct = paste0(round(100*estimate/ctrlmean), "%")) %>% 
                       filter(term == varname & ctrl == ctrlname), 
                     aes(x = yvar, y = estimate, fill = gender, color = gender)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.4) +
    scale_fill_manual(values = c(col1, col2)) +
    scale_color_manual(values = c(col1, col2)) +
    labs(y = ylab, x = "",
         fill = "", color = "") +
    geom_errorbar(aes(ymin = ylb, ymax = yub, width = 0.1, color = gender),
                  position = position_dodge(width = 0.9), show.legend = FALSE) + 
    geom_hline(yintercept = 0, color = "black", alpha = 0.5) +
    geom_text(aes(label = pct, y = 0), position = position_dodge(width = 0.9), vjust = -0.5, show.legend = FALSE) + 
    theme_minimal() +
    theme(axis.text.x = element_text(angle = angle, vjust = 0.5),
          #legend.position = "bottom", 
          text = element_text(size = 18),
          axis.text = element_text(size = 11))
  if (ecls){
    plot_out <- plot_out + ylim(-0.5,0.4)
  }
  print(plot_out)
  ggsave(glue("{out}/{filename}.png"), plot_out + ylim(-0.11, 0.132), width = 9, height = 5)
}
