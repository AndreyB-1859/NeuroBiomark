# ============================================================================
# main.py - WITH MENU SYSTEM
# ============================================================================

from src.process_data.load_data import prepare_group_folds
from src.models.utils import run_folds, anova_test_multi_class
from config.config import config
from src.models.augmentation_evaluation_CNN import run_augmentation_evaluation

device = config.device

# ============================================================================
# MENU SYSTEM
# ============================================================================

#EfficientNetB0 option
# def print_menu():
#     print("\n" + "="*80)
#     print("NEURO BIOMARKER ALS PROJECT - MAIN MENU")
#     print("="*80)
#     print("\nSelect Mode:")
#     print("1. Train DenseNet121 (Your Original Setup)")
#     print("2. Evaluate Augmentations - Quick Test (2 augmentations, ~1-2 hours)")
#     print("3. Evaluate Augmentations - Full (All augmentations, ~10-20 hours)")
#     print("4. View Augmentation Results (if evaluation already done)")
#     print("0. Exit")
#     print("="*80)

#BasicCNN option
def print_menu():
    print("\n" + "="*80)
    print("NEURO BIOMARKER ALS PROJECT - MAIN MENU")
    print("="*80)
    print("\nSelect Mode:")
    print("1. Train DenseNet121 (Original Setup)")
    print("2. BasicCNN - Quick Test (3 augmentations, ~1 hour)")
    print("3. BasicCNN - Full Evaluation (22 augmentations, ~10-15 hours)")
    print("4. EfficientNetB0 - Quick Test (3 augmentations, ~1 hour)")
    print("5. EfficientNetB0 - Full Evaluation (22 augmentations, ~10-15 hours)")
    print("6. View Results")
    print("0. Exit")
    print("="*80)

def run_original_training():
    """Your original DenseNet121 training"""
    print("\n" + "="*80)
    print("MODE: TRAINING WITH DENSENET121")
    print("="*80 + "\n")
    
    prepare_group_folds()
    
    for fold in range(config.no_of_folds):
        config.fold_no = fold
        print(f"\nFold: {fold}")
        run_folds(fold)
    
    anova_test_multi_class(config.no_of_folds)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80 + "\n")

#EfficientNetB0 option for data aug eval
def run_augmentation_eval(quick=True):
    """Run augmentation evaluation"""
    
    prepare_group_folds()

    try:
        from src.models.augmentation_evaluation import run_augmentation_evaluation
        run_augmentation_evaluation(quick_test=quick)
    except ImportError as e:
        print(f"\n❌ Error importing augmentation evaluation: {e}")
        print("Make sure augmentation_evaluation.py is in src/models/")
    except Exception as e:
        print(f"\n❌ Error during augmentation evaluation: {e}")
        import traceback
        traceback.print_exc()

#BasicCNN option for data aug eval
def run_basic_cnn_augmentation_eval(quick=True):
    """Run BasicCNN augmentation evaluation"""
    print("\n" + "="*80)
    print("MODE: BASICCNN AUGMENTATION EVALUATION")
    print("="*80 + "\n")
    
    # Prepare folds (5 group folds)
    prepare_group_folds()
    
    # Run evaluation
    try:
        results_df, best_fold_logs = run_augmentation_evaluation(quick_test=quick)
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
        print(f"\nBest augmentation: {results_df.iloc[0]['augmentation']}")
        print(f"Best accuracy: {results_df.iloc[0]['mean_val_acc']:.4f}")
        print(f"\nResults saved to: logs/augmentation_evaluation/")
        print("="*80 + "\n")
        
    except ImportError as e:
        print(f"\n❌ Error importing: {e}")
        print("Make sure all files are placed correctly")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def view_results():
    """View augmentation evaluation results"""
    import os
    results_path = os.path.join(config.logs_dir_path, 'augmentation_evaluation', 'augmentation_comparison.csv')
    
    if not os.path.exists(results_path):
        print(f"\n❌ Results not found at: {results_path}")
        print("Run augmentation evaluation first (Option 2 or 3)")
        return
    
    import pandas as pd
    df = pd.read_csv(results_path)
    
    print("\n" + "="*80)
    print("AUGMENTATION EVALUATION RESULTS")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        
        if choice == "0":
            print("\nExiting...")
            break
        
        elif choice == "1":
            run_original_training()

        elif choice == "2":
            print("\n⚡ Quick test: 3 augmentations × 5 folds = ~1 hour")
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                run_basic_cnn_augmentation_eval(quick=True)
        
        elif choice == "3":
            print("\n⏳ Full evaluation: 22 augmentations × 5 folds = ~10-15 hours")
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                run_basic_cnn_augmentation_eval(quick=False)
        
        elif choice == "4":
            print("\n⚠️ This will take 1-2 hours with GPU")
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                run_augmentation_eval(quick=True)
        
        elif choice == "5":
            print("\n⚠️ This will take 10-20 hours with GPU")
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                run_augmentation_eval(quick=False)
        
        elif choice == "6":
            view_results()
        
        else:
            print("\n❌ Invalid choice. Please select 0-4.")


if __name__ == "__main__":
    main()






# # from src.process_data.load_data import prepare_folds
# from src.process_data.load_data import prepare_group_folds
# # from src.process_data.load_data import prepare_leave_one_group_out_folds
# from src.models.utils import run_folds
# # from src.models.utils import anova_test_per_fold
# from src.models.utils import anova_test_multi_class
# from config.config import config
# # from src.models.augmentation_evaluation import run_augmentation_evaluation

# device = config.device

# prepare_group_folds()

# for fold in range(config.no_of_folds):

#     config.fold_no = fold
#     print(f"\nFold: {fold}")
#     run_folds(fold)

# anova_test_multi_class(config.no_of_folds)



