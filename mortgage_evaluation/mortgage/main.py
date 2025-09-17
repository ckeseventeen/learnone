import mortgage
if __name__=="__main__":
    obj=mortgage.mortgage()
    if obj.parameters['run_mod'] == 0:
        obj.get_weight()
    if obj.parameters['run_mod'] == 1:
        obj.data_hive_test()
    if obj.parameters['run_mod'] == 2:
        obj.mort_analysis()
    if obj.parameters['run_mod'] == 3:
        obj.RNNwithoutMultiprocess()










