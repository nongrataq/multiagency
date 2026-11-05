def quiz_radios(app):
    return [radio for radio in app.radio if radio.key != 'page']


def select_page(app, name):
    next(radio for radio in app.radio if radio.key == 'page').set_value(name).run()
    return app


def select_activity(app, name):
    next(select for select in app.selectbox if select.key == 'activity').select(name).run()
    return app


def click_button(app, label):
    next(button for button in app.button if button.label == label).click().run()
    return app
