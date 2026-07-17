async def test_ocr_document_upload_returns_extracted_entities(client):
    response = await client.post(
        "/documents/ocr",
        files={"file": ("form.png", b"irrelevant-bytes", "image/png")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "form.png"
    assert body["status"] == "clean"
    assert body["extracted_entities"]["patient_name"] == "Jane Doe"


async def test_uploaded_document_appears_in_list(client):
    await client.post(
        "/documents/ocr",
        files={"file": ("form.png", b"irrelevant-bytes", "image/png")},
    )

    response = await client.get("/documents")

    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 1
    assert documents[0]["filename"] == "form.png"
