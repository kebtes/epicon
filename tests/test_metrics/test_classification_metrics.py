from epicon.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


class TestAccuracyScore:
    def test_perfect(self):
        assert accuracy_score([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0

    def test_partial(self):
        assert accuracy_score([0, 1, 0, 1], [0, 1, 1, 1]) == 0.75

    def test_all_wrong(self):
        assert accuracy_score([0, 0, 0], [1, 1, 1]) == 0.0


class TestConfusionMatrix:
    def test_binary(self):
        cm = confusion_matrix([0, 1, 0, 1], [0, 1, 1, 1])
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1
        assert cm[1, 1] == 2
        assert cm[0, 1] == 1
        assert cm[1, 0] == 0


class TestPrecisionScore:
    def test_binary(self):
        prec = precision_score([0, 1, 0, 1], [0, 1, 1, 1])
        assert prec == 2.0 / 3.0

    def test_perfect(self):
        assert precision_score([0, 1], [0, 1]) == 1.0


class TestRecallScore:
    def test_binary(self):
        rec = recall_score([1, 0, 1, 1], [1, 0, 0, 1])
        assert rec == 2.0 / 3.0


class TestF1Score:
    def test_binary(self):
        f1 = f1_score([0, 1, 0, 1], [0, 1, 1, 1])
        assert f1 > 0
        assert f1 <= 1.0
