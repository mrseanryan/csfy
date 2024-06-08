http_codes = {
    401: {
        "description": "Error: Not authenticated",
        "content": {
            "application/json": {
                "example": {"detail": "Not authenticated - Missing or incorrect credentials"},
            }
        },
    },
    403: {
        "description": "Error: Forbidden",
        "content": {
            "application/json": {
                "example": {"detail": "Forbidden"},
            }
        },
    },
}
