install.packages('pacman')
pacman::p_load(readxl, haven, tidyr, dplyr, mirt, knitr, sjPlot, sjmisc, openxlsx)
setwd('D:/repos/pessoalredem-index')
# ASIL
dados <- read_excel('./df_res.xlsx')

dadosTRI <- dados %>%
  select(tempo_atuacao_percent_cum,
    mesa_1_cum,
    mesa_2_cum,
    relatorias_ln_cum,
    pos_lider_cum,
    pos_comiss_pr_cum,
    mand_dep_estadual_cum,
    mand_ver_cum,
    mand_sen_cum,
    fid_municipais_cum,
    fid_gerais_cum,
  )
dadosTRI <- dadosTRI %>% drop_na()

dados_ordinal <- dadosTRI %>%
  mutate(across(everything(), ~{
    if(is.numeric(.x)) {
      as.integer(cut(.x, breaks = 5, labels = FALSE))  # Converte para inteiros
    } else {
      as.integer(factor(.x, ordered = TRUE))
    }
  }))
dadosTRI <- dados_ordinal %>% drop_na()
head(dados_ordinal)

mod <- (mirt(
  dadosTRI, 
  model='
    Factor1 = 1,2,3,4,5,6
    Factor2 = 7,8
    Factor3 = 10,11', 
    verbose = FALSE, 
    itemtype = 'graded', 
    SE = TRUE
  ))
M2(mod, type = "C2", calcNULL = FALSE)
itemfit(mod)

# IRT parameters
IRT <- coef(mod, IRTpars = TRUE, simplify = TRUE)
IRT$items
summary(mod)

#Salva scores thetas da TRI 
theta <- fscores(mod)

# calcule o valor mínimo e máximo de theta
min_theta <- min(theta)
max_theta <- max(theta)

# normalize os valores theta na escala de 0 a 10
indice <- 10 * (theta - min_theta) / (max_theta - min_theta)

# arredonde o índice para duas casas decimais
indice <- round(indice, 2)

#junte os valores ao banco
res <- cbind(dados, indice)
head(res, 10)

summary(res$F1)

write.xlsx(res, "TRI_res.xlsx")

