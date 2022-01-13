import logging
from typing import List
from functools import partial

import numpy as np
import pandas as pd

from ddf_utils.str import to_concept_id
from ddf_utils.transformer import translate_column as tc
from ddf_utils.chef.helpers import debuggable, build_dictionary, read_opt
from ddf_utils.chef.model.ingredient import (Ingredient,
                                             get_ingredient_class,
                                             ConceptIngredient,
                                             DataPointIngredient)
from ddf_utils.chef.model.chef import Chef


logger = logging.getLogger('custom_procedure')


@debuggable
def rolling_average(chef: Chef, ingredients: List[DataPointIngredient], result,
                    days, not_calculate) -> DataPointIngredient:
    ingredient = ingredients[0]
    logger.info("calculating {} days averages for {}".format(days, ingredient.id))
    di = ingredient.compute()
    keys = ingredient.key
    new_data = dict()

    for k, df in di.items():
        if not (('smoothed' in k) or (k in not_calculate)):
            k_new = k + '_smoothed'
            df_ = df.copy()
            logger.info(k)
            df_[k_new] = df_.groupby(keys)[k].transform(lambda x: x.rolling(days).mean())
            df_ = df_.drop(columns=[k])
            new_data[k_new] = df_

    return DataPointIngredient.from_procedure_result(result, ingredient.key, data_computed=new_data)


@debuggable
def gen_concept(chef: Chef, ingredients: List[DataPointIngredient], result, base) -> ConceptIngredient:
    ingredient = ingredients[0]
    logger.info("generate concepts for {}".format(ingredient.id))

    di = ingredient.get_data()
    cb = chef.dag.get_node(base).evaluate().get_data()['concept'].set_index('concept')

    indicators = list(di.keys())  # this one should contain only the smoothed indicators
    # create a list of indicator ids without the tailing _smoothed
    indicators_ = list(map(lambda x: x[:-9], indicators))
    # get names/tags/descriptions for them
    names = cb.loc[indicators_, "name"].map(lambda x: x + " (7 days smoothed)")
    tags = cb.loc[indicators_, 'tags'].map(lambda x: x + "_smoothed")
    desc = cb.loc[indicators_, 'description']

    df = pd.DataFrame({'concept': indicators, 'name': names, 'description': desc, 'tags': tags})
    df = df.set_index('concept')
    df['concept_type'] = 'measure'

    # also move some of already smoothed indicators to smoothed tags
    cb.loc[cb.index.str.contains('smoothed'), "tags"] = cb.loc[cb.index.str.contains('smoothed'), "tags"].map(lambda x: x + "_smoothed")
    res = cb.append(df)

    new_data = {'concept': res.reset_index()}

    return ConceptIngredient.from_procedure_result(result, 'concept', data_computed=new_data)
