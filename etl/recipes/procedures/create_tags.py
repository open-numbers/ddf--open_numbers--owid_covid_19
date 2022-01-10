import logging
from typing import List
from functools import partial

import numpy as np
import pandas as pd

from ddf_utils.str import to_concept_id
from ddf_utils.transformer import translate_column as tc
from ddf_utils.chef.helpers import debuggable, build_dictionary, read_opt
from ddf_utils.chef.model.ingredient import Ingredient, get_ingredient_class, ConceptIngredient, EntityIngredient
from ddf_utils.chef.model.chef import Chef


logger = logging.getLogger('custom_procedure')


@debuggable
def in_concept(chef: Chef, ingredients: List[ConceptIngredient], result,
               input_column, out_column) -> ConceptIngredient:
    ingredient = ingredients[0]
    logger.info("create tags concept property from {}: column {}".format(ingredient.id, input_column))
    di = ingredient.get_data()
    new_data = dict()
    new_data['concept'] = di['concept'].copy()
    new_data['concept'][out_column] = new_data['concept'][input_column].map(to_concept_id)

    return ConceptIngredient.from_procedure_result(result, ingredient.key, data_computed=new_data)


@debuggable
def in_entity(chef: Chef, ingredients: List[ConceptIngredient], result,
              input_column) -> EntityIngredient:
    ingredient = ingredients[0]
    logger.info("create tags entity from {}: column {}".format(ingredient.id, input_column))
    df = ingredient.get_data()['concept']
    new_data = dict()
    names = df[input_column].dropna().drop_duplicates()
    tags = names.map(to_concept_id)
    res = pd.DataFrame({'tag': tags.values, 'name': names.values})
    res['parent'] = np.nan
    new_data['tag'] = res

    return EntityIngredient.from_procedure_result(result, 'tag', data_computed=new_data)
