import pandas as pd
import numpy as np

df_pubs = pd.read_excel("./2023_Completo_redem_0304.xlsx")
df = pd.read_csv("df_res.csv")

print(f"Total de mensagens: {len(df)}")

df[df["min_distance"] > 0].sort_values("min_distance").reset_index().merge(
    df_pubs[["Message-ID", "Profile", "partido", "bloco"]],
    left_on="pub_id",
    right_on="Message-ID",
    how="left",
)

df['is_fn'] = df['min_distance'].apply(lambda x: 1 if x < 0.3 else 0)


df_rank = (
    df.merge(df_pubs, left_on="pub_id", right_on="Message-ID", how="left")
    .groupby("Profile")
    .agg({"partido": "first", "pub_id": "count", "is_fn": "sum", "min_distance": "min"})
    .sort_values("min_distance")
)

df.to_excel("5_df_with_msgs.xlsx", index=False)
df_rank.to_excel("5_df_rank.xlsx", index=False)

