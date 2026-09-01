import pytest
from heatwave_analysis.association_models import assert_no_target_leakage
from heatwave_analysis.variable_dictionary import PRIMARY_ASSOCIATION_PREDICTORS

def test_primary_predictors_are_leakage_free(): assert_no_target_leakage(PRIMARY_ASSOCIATION_PREDICTORS)
def test_current_temperature_is_rejected():
    with pytest.raises(ValueError): assert_no_target_leakage(["tmax"])
