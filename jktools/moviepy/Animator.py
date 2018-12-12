class Animator:
    def __init__(self, variables):
        self.variables = variables
        self.animations = []

    def add_animations(self, *animations):
        for animation in animations:
            self.animations.append(animation)

    def update(self, t):
        for animation in self.animations:
            animation.update(self.variables, t)
