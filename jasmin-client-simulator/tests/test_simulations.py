def test_simulation_lifecycle(client):
    # 1. Fetch available profiles
    profiles_resp = client.get("/api/v1/profiles")
    profile = profiles_resp.json()[0]
    profile_id = profile["id"]

    # 2. Start simulation
    sim_resp = client.post("/api/v1/simulations", json={
        "profile_id": profile_id,
        "scenario_title": "Integration Test Simulation"
    })
    assert sim_resp.status_code == 201
    sim_data = sim_resp.json()
    sim_id = sim_data["id"]
    assert sim_data["status"] == "active"
    assert sim_data["current_stage"] == "greetings"

    # 3. Send first sales rep message
    turn1_resp = client.post(f"/api/v1/simulations/{sim_id}/messages", json={
        "message": "Hello Carlos! Welcome to our travel agency. How can I help you today?"
    })
    assert turn1_resp.status_code == 200
    t1_data = turn1_resp.json()
    assert "customer_response" in t1_data
    assert t1_data["current_stage"] == "needs_discovery"

    # 4. Fetch history
    hist_resp = client.get(f"/api/v1/simulations/{sim_id}/history")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history) == 3 # 1 initial customer greeting + 1 rep msg + 1 customer reply
    assert history[0]["sender"] == "customer"
    assert history[1]["sender"] == "sales_rep"
    assert history[2]["sender"] == "customer"

    # 5. End simulation
    end_resp = client.post(f"/api/v1/simulations/{sim_id}/end")
    assert end_resp.status_code == 200
    assert end_resp.json()["status"] == "completed"
