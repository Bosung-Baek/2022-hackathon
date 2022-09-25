import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)    # 분당 말하기 속도

voice = engine.getProperty('voices')
engine.setProperty('voice', voice[1].id)    # 영어로 나옴


def say(object, direc):
    text = f"There's {object} on your {direc}."

    engine.say(text)
    engine.runAndWait()
    # engine.stop()


if __name__ == '__main__':
    say('car', 'right')
    say('car', 'center down')
