import numpy as np

from epicon.preprocessing import LabelEncoder, OneHotEncoder


class TestLabelEncoder:
    def test_fit_transform(self):
        encoder = LabelEncoder()
        result = encoder.fit_transform(["cat", "dog", "bird", "dog"])
        # np.unique returns sorted: ['bird', 'cat', 'dog']
        np.testing.assert_array_equal(result, [1, 2, 0, 2])

    def test_inverse_transform(self):
        encoder = LabelEncoder()
        encoder.fit(["cat", "dog", "bird"])
        # np.unique returns sorted: ['bird', 'cat', 'dog']
        result = encoder.inverse_transform([0, 1, 2])
        np.testing.assert_array_equal(result, ["bird", "cat", "dog"])

    def test_classes(self):
        encoder = LabelEncoder()
        encoder.fit([3, 1, 2])
        np.testing.assert_array_equal(encoder.classes_, [1, 2, 3])


class TestOneHotEncoder:
    def test_fit_transform(self):
        encoder = OneHotEncoder()
        result = encoder.fit_transform(np.array([["cat"], ["dog"], ["bird"]]))
        np.testing.assert_array_equal(result, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def test_inverse_transform(self):
        encoder = OneHotEncoder()
        encoder.fit(np.array([["cat"], ["dog"], ["bird"]]))
        result = encoder.inverse_transform(np.array([[0, 1, 0], [0, 0, 1]]))
        np.testing.assert_array_equal(result, [["cat"], ["dog"]])
