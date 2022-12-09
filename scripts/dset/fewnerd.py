import typer

app = typer.Typer()


def main(
    src: str,
    target: str = typer.Option(...)
):
    """将fewnerd的json数据转换为jsonl

    Args:
        src (str): _description_
        target (str, optional): _description_. Defaults to typer.Option(...).
    """
    pass


if __name__ == "__main__":
    app()
