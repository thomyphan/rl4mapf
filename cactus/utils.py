import numpy
import random

def assertTensorEquals(first, second):
    assert (first == second).all(), "Expected {}, got {}".format(first, second)

def assertEquals(first, second):
    assert first == second, "Expected {}, got {}".format(first, second)

def assertTrue(first):
    assert first, "Expected to be True"

def assertFalse(first):
    assert not first, "Expected to be False"

def assertAtLeast(first, second):
    assert first >= second, "Expected {} >= {}".format(first, second)

def assertGreater(first, second):
    assert first > second, "Expected {} > {}".format(first, second)

def assertAtMost(first, second):
    assert first <= second, "Expected {} <= {}".format(first, second)

def assertLess(first, second):
    assert first < second, "Expected {} < {}".format(first, second)

def assertContains(collection, element):
    assert element in collection, "Expected element in {}, got {}".format(collection, element)

def get_param_or_default(params, label, default_value=None):
    if label in params:
        return params[label]
    else:
        return default_value

def argmax(values):
    max_value = max(values)
    default_index = numpy.argmax(values)
    candidate_indices = []
    for i,value in enumerate(values):
        if value >= max_value:
            candidate_indices.append(i)
    if not candidate_indices:
        return default_index
    return random.choice(candidate_indices)