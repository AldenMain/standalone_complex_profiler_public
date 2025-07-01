import pandas as pd
from emergent_agency_index import compute_emergent_agency_index

df = pd.DataFrame({
    'post_text': [
        "I don't know if I did the right thing.",
        "Sometimes I wonder if there's meaning to all this.",
        "Just another day, just another post.",
        "I regret what I said, but I meant it in the moment."
    ]
})

eai_df = compute_emergent_agency_index(df['post_text'].tolist())

print(eai_df)

