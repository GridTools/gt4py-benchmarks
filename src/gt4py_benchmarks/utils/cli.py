import ast

import click


class KeyValueArg(click.ParamType):
    name = "key-value"

    def convert(self, value, param, ctx):
        try:
            key, value = value.split("=", 1)
        except ValueError:
            self.fail(f"expected key-value pair, got {value}", param, ctx)

        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass

        return key, value
