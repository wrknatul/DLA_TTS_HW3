import hw_tts.text as text


def get_data(user_tests=None):
    tests = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    if user_tests is not None:
        tests = user_tests
    data_list = list(text.text_to_sequence(test, ['english_cleaners']) for test in tests)

    return data_list
