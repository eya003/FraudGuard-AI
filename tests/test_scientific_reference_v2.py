"""The presentation may not drift from the saved V2 notebook outputs."""
import unittest
from backend.services.comparison_service import model_comparison_history
from frontend.model_reference import load_model_reference

class V2ScientificCatalogTest(unittest.TestCase):
    def test_five_families_and_thirteen_variants(self):
        entries=load_model_reference()['variants']
        self.assertEqual(len(entries),13)
        self.assertEqual(len(set(e['family'] for e in entries)),5)
        self.assertTrue(all(e['pr_auc']>=0 and e['pr_auc']<=1 for e in entries))
        self.assertTrue(all(e['roc_auc']>=0 and e['roc_auc']<=1 for e in entries))
        self.assertTrue(all(e['notebook_cell']>0 for e in entries))

    def test_no_old_xgboost_and_rf_is_only_executable_model(self):
        entries=load_model_reference()['variants']
        self.assertFalse(any('V1' in e['family'] for e in entries))
        self.assertEqual([e['id'] for e in entries if e['inference_available']],['rf_opt'])
        history=model_comparison_history()['items']
        self.assertEqual(len(history),13)
        self.assertTrue(all(e['split']=='Validation' for e in history))

    def test_operational_selection_is_based_on_validation_cost(self):
        data=load_model_reference()
        costs=data['business_reference']['results']
        rf=next(x for x in costs if x['id']=='rf_opt')
        self.assertLess(rf['total_cost'],min(x['total_cost'] for x in costs if x['id']!='rf_opt'))
        self.assertEqual(rf['alerts'],1997)
        self.assertEqual(rf['detected_frauds'],1581)

    def test_test_confusion_and_cost_identity(self):
        final=load_model_reference()['test_final']
        self.assertEqual(final['tn']+final['fp']+final['fn']+final['tp'],final['rows'])
        self.assertEqual(final['tp']+final['fp'],final['alerts'])
        self.assertAlmostEqual(final['total_cost'],final['missed_fraud_amount']+final['alert_cost'],places=2)
        self.assertEqual(final['fn'],208)


    def test_notebook_curve_and_eda_assets_are_present(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        self.assertTrue((root/'notebooks'/'fraude-v2.ipynb').is_file())
        self.assertTrue((root/'docs'/'figures'/'precision_recall_validation_v2.png').is_file())
        self.assertTrue((root/'data'/'dataset'/'eda_aggregates_v1'/'fraudguard_eda_aggregates_v1.json').is_file())

    def test_all_budget_metrics_from_notebook_are_present(self):
        data = load_model_reference()
        ranking = data['precision_at_k']
        self.assertEqual(len(ranking), 18)
        self.assertEqual({row['k'] for row in ranking}, {1000, 3000, 5000})
        self.assertEqual(len({row['model_id'] for row in ranking}), 6)
        self.assertEqual(len({(row['model_id'], row['k']) for row in ranking}), 18)
        self.assertEqual(next(r['precision'] for r in ranking if r['model_id'] == 'autoencoder' and r['k'] == 5000), 0.1852)
        self.assertEqual(next(r['recall'] for r in ranking if r['model_id'] == 'isolation' and r['k'] == 5000), 0.3598)

    def test_saved_daily_budget_scenarios_are_complete_and_costs_consistent(self):
        data = load_model_reference()
        for key, expected_count, expected_models in [('daily_topk_reference', 18, 6),
                                                     ('daily_threshold_reference', 9, 3)]:
            ref = data[key]
            cases = ref['results']
            self.assertEqual(len(cases), expected_count)
            self.assertEqual(len({case['model_id'] for case in cases}), expected_models)
            self.assertEqual({case['daily_budget'] for case in cases}, {10, 20, 30})
            self.assertEqual(len({(case['model_id'], case['daily_budget']) for case in cases}), expected_count)
            for case in cases:
                self.assertAlmostEqual(case['total_cost'], case['missed_fraud_amount'] + 5 * case['alerts'], places=2)
        rf = next(r for r in data['daily_topk_reference']['results'] if r['model_id'] == 'rf_opt' and r['daily_budget'] == 30)
        self.assertEqual((rf['alerts'], rf['detected_frauds']), (4230, 1734))

    def test_nine_page_navigation_without_obsolete_ui_sections(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        app=(root/'frontend'/'app.py').read_text(encoding='utf-8')
        self.assertEqual(app.count('st.Page('),9)
        for name in ['about.py','data_explorer.py','import_transactions.py','ml_laboratory.py','alert_center.py']:
            self.assertNotIn(f'pages/{name}',app)

if __name__=='__main__': unittest.main()
