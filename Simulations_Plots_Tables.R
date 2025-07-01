## This script generates plots and tables based on the simulation results ##


## Conley et al (2008) simulation results ##
ss = c(0.5, 1, 1.5)
res_conley = readRDS("Results_Conley.RDS")

library(ggplot2)

d_plot = Map(function(d, s) data.frame(cbind(d, s), Method = rownames(d)), res_conley, ss) |>
  do.call(rbind, args = _) |>
  tidyr::pivot_longer(cols = c("MAE", "Coverage", "MIL"))
d_plot$name = factor(d_plot$name, levels = c("MAE", "Coverage", "MIL"))

p = ggplot(d_plot, aes(x = s, y = value, colour = Method)) +
  geom_line(alpha = 0.8) +
  geom_point(size = 2, alpha = 0.8) +
  facet_wrap(~name, scales = "free") +
  scale_colour_viridis_d() +
  labs(y = "", x = "Instrument Strength (s)") + 
  theme_bw()

ggsave("Simulation_Conley_Results.pdf", plot = p, width = 8, height = 4)
