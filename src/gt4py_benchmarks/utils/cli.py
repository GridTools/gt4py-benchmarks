import itertools
import re

import click
import typing_extensions

from ..runtime import runtimes
from ..numerics import stencil_backends


def cli_name(name):
    return re.sub("([a-z])([A-Z])", r"\1-\2", name.replace("_", "-")).lower()


def per_runtime_cli(func, options=None, **defaults):
    @click.group()
    def cli():
        pass

    for runtime in runtimes.REGISTRY:

        @cli.group(name=cli_name(runtime.__name__.replace("Runtime", "")))
        def group():
            pass

        for stencil_backend in stencil_backends.REGISTRY:

            def command(runtime=runtime, stencil_backend=stencil_backend, **kwargs):
                runtime_kwargs = {k: v for k, v in kwargs.items() if k in runtime.__fields__}
                stencil_backend_kwargs = {
                    k: v for k, v in kwargs.items() if k in stencil_backend.__fields__
                }
                option_args = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in runtime.__fields__ and k not in stencil_backend.__fields__
                }
                sb = stencil_backend(**stencil_backend_kwargs)
                rt = runtime(stencil_backend=sb, **stencil_backend_kwargs)
                func(rt, **option_args)

            for k, v in itertools.chain(
                runtime.__fields__.items(), stencil_backend.__fields__.items()
            ):
                if k != "stencil_backend":
                    dtype = None
                    if v.type_ is int:
                        dtype = int
                    elif v.type_ is float:
                        dtype = float
                    elif typing_extensions.get_origin(v.type_) is typing_extensions.Literal:
                        dtype = click.Choice(typing_extensions.get_args(v.type_))

                    default = defaults.get(k, v.default)

                    click.option(
                        f"--{cli_name(k)}",
                        type=dtype,
                        default=default,
                        required=default is None,
                        show_default=True,
                    )(command)

            if options:
                for option in options:
                    option(command)

            group.command(name=cli_name(stencil_backend.__name__.replace("StencilBackend", "")))(
                command
            )

    return cli
