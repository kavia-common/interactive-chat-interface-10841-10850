import json
import os

from src.api.main import app

if __name__ == "__main__":
    # Generate OpenAPI schema from the running app
    openapi_schema = app.openapi()

    # Ensure interfaces directory exists at container root
    output_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "interfaces"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "openapi.json")

    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    print(f"OpenAPI schema written to: {output_path}")
