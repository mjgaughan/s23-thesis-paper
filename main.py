from get_data_embeddings import get_openai_embeddings
from run_models import linear_ots_class
#from zero_shot import zero_shot_class

def main():
    # data is first cleaned in clean_data.py, no need to do that again
    # data is then combined, if necessary, in append_data.py
    clean_data_path = "../../nobu-data/s23-thesis-paper/0_lkernel_params_mutable.csv"
    embeddings_filepath = "../../nobu-data/s23-thesis-paper/52423_test500_0_lkf_param_embeddings.csv"
    get_openai_embeddings(clean_data_path, embeddings_filepath)
    #run linear models
    linear_ots_class(embeddings_filepath)


main()