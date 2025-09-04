import pandas as pd

def get_3ct_interaction():
    ct1, ct2, ct3, ct4 = "4", "1", "2", "0"
    interaction_df = pd.DataFrame(
        [
            {
                "receptor_cell": ct1,
                "receptor_subtype": f"{ct1}_sub0",
                "interaction_type": "neutral",
            },
            {
                "receptor_cell": ct2,
                "receptor_subtype": f"{ct2}_sub0",
                "interaction_type": "neutral",
            },
            {
                "receptor_cell": ct3,
                "receptor_subtype": f"{ct3}_sub0",
                "interaction_type": "neutral",
            },
            {
                "receptor_cell": ct2,
                "sender_cell": ct1,
                "receptor_subtype": f"{ct2}_sub1",
                "radius_of_effect": 20,
                "interaction_type": "interaction",
            },
            {
                "receptor_cell": ct3,
                "sender_cell": ct2,
                "receptor_subtype": f"{ct3}_sub1",
                "radius_of_effect": 10,
                "interaction_type": "interaction",
            },
            {
                "receptor_cell": ct4,
                "receptor_subtype": f"{ct4}_sub0",
                "interaction_type": "neutral",
            },
        ]
    )

    return interaction_df