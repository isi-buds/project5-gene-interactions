# Results-df.py
import pandas as pd

results = [['Random Forest', 'firstset', 'unreduced', 3.3081, 0.1366],
           ['Random Forest', 'firstset', 'reduced', 1.1979, 0.5186],
           ['Random Forest', 'firstset measures', 'unreduced', 2.5730, 0.2499],
           ['Random Forest', 'firstset measures', 'reduced', 0.7970, 0.6737],
           ['Multinomial logistic', 'firstset', 'unreduced', 4.1114, 0.0570],
           ['Multinomial logistic', 'firstset', 'reduced', 1.7354, 0.0570],
           ['Multinomial logistic', 'firstset measures', 'unreduced', 3.4440, 0.1048],
           ['Multinomial logistic', 'firstset measures', 'reduced', 1.2537, 0.1121],
           ['Multinomial logistic', 'secondset', 'reduced', 1.3719, 0.4534],
           ['Multinomial logistic', 'secondset prob', 'reduced', 1.1397, 0.5100],
           ['Multinomial logistic', 'secondset prob measures', 'reduced', 0.5118, 0.8912]]

cols = ['Method', 'X', 'y', 'Log loss', 'Accuracy']

results_df = pd.DataFrame(results, columns = cols)

ml_results = results_df[results_df['Method'] == 'Multinomial logistic']

rf_results = results_df[results_df['Method'] == 'Random Forest']

