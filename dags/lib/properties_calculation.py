from rdkit.Chem import Descriptors

DESCRIPTOR_FUNCTIONS = {
    "tpsa": Descriptors.TPSA,
    "logp": Descriptors.MolLogP,
    "mw": Descriptors.MolWt,
    "h_acceptors": Descriptors.NumHAcceptors,
    "h_donors": Descriptors.NumHDonors,
    "ring_count": Descriptors.RingCount,
}


class PropertiesCalculator:
    def calculate_properties(self, df):
        df = df.copy()

        self._add_descriptors(df)
        self._add_rankings(df)
        # self._add_labels(df)

        return df

    def _add_descriptors(self, df):
        for column_name, descriptor_function in DESCRIPTOR_FUNCTIONS.items():
            df[column_name] = df["mol"].apply(descriptor_function)

    def _add_rankings(self, df):
        descriptor_columns = DESCRIPTOR_FUNCTIONS.keys()

        for column_name in descriptor_columns:
            df[f"{column_name}_ranked"] = (
                df[column_name]
                .rank(pct=True)
            )

    # def _add_labels(self, df):
    #     df["label"] = (
    #         df["smiles"]
    #         + '__<a href="https://www.drugbank.ca/drugs/'
    #         + df["molecule_id"]
    #         + '" target="_blank">'
    #         + df["molecule_id"]
    #         + "</a>__"
    #         + df["name"]
    #     ).str.replace("'", "", regex=False)
