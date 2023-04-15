from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData

def attack(loss_train,loss_test):
    attacks_result = mia.run_attacks(
        AttackInputData(
            loss_train = loss_train,
            loss_test = loss_test))
    return attacks_result.single_attack_results[0].get_auc()
    
if __name__ == "__main__":
    pass