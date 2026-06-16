from app import _form_payload


class FakeForm(dict):
    pass


def test_form_payload_casts_numeric_fields():
    form = FakeForm(
        area="6500",
        bedrooms="3",
        bathrooms="2",
        stories="2",
        parking="1",
        age="10",
        mainroad="yes",
        guestroom="no",
        basement="yes",
        hotwaterheating="no",
        airconditioning="yes",
        prefarea="yes",
        furnishingstatus="semi-furnished",
        city_zone="suburban",
    )

    payload = _form_payload(form)

    assert payload["area"] == 6500.0
    assert payload["bedrooms"] == 3.0
    assert payload["city_zone"] == "suburban"
