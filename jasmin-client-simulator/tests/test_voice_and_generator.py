import io

def test_generate_persona_archetypes(client):
    for archetype in ["satisfied", "unsatisfied", "hesitant", "anxious", "demanding", "rushed", "family"]:
        response = client.post(f"/api/v1/profiles/generate?archetype={archetype}&destination=Djerba&budget=3000")
        assert response.status_code == 201
        data = response.json()
        assert data["archetype"] == archetype
        assert "Djerba" in data["destination_preferences"]

def test_send_voice_message(client):
    # 1. Fetch profile
    res_prof = client.get("/api/v1/profiles")
    prof_id = res_prof.json()[0]["id"]
    
    # 2. Start simulation
    res_sim = client.post("/api/v1/simulations", json={"profile_id": prof_id, "scenario_title": "Voice Test"})
    sim_id = res_sim.json()["id"]

    # 3. Post mock voice audio file
    fake_audio_bytes = b"RIFF....WAVEfmt ....data...."
    files = {"file": ("recording.webm", io.BytesIO(fake_audio_bytes), "audio/webm")}
    
    res_voice = client.post(f"/api/v1/simulations/{sim_id}/voice_message", files=files)
    assert res_voice.status_code == 200
    voice_data = res_voice.json()
    assert "customer_response" in voice_data
    assert "trust_level" in voice_data
    assert "audio_response" in voice_data
