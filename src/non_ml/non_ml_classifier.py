class NonMLClassifier:
    def __init__(self):
        pass

    def predict(self, data):
        results = []

        for idx, row in data.iterrows():
            if row['Primary_applicant_age_in_years'] < 35 and row['Housing_own'] == 1 and (
                    row['Savings_Medium'] == 1 or row['Savings_Very high'] == 1) and (
                    row['Loan_history_existing loans paid back duly till now'] == 1 or row[
                'Loan_history_critical/pending loans at other banks'] == 1):
                results.append(0)
            else:
                results.append(1)

        return results
