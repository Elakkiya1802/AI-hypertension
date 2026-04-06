const API_URL = "https://ai-hypertension.onrender.com";

async function register(){
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    const res = await fetch(API_URL + "/register", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            email: email,
            password: password,
            role: "user"
        })
    });

    const data = await res.json();

    console.log("STATUS:", res.status);
    console.log("RESPONSE:", data);

    if (!res.ok) {
        alert(data.detail || JSON.stringify(data));
        return;
    }

    alert(data.message || "Registered successfully");
    window.location.href = "index.html";
}


async function login() {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    const res = await fetch(API_URL + "/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ email, password })
    });

    const data = await res.json();

    if (!res.ok) {
        alert(data.detail || "Login failed");
        return;
    }

    localStorage.setItem("user_id", data.user_id);
    localStorage.setItem("role", data.role);

    if (data.role === "admin") {
        window.location.href = "admin.html";
    } else {
        window.location.href = "user.html";
    }
}

/* ---------- USER HISTORY ---------- */

async function loadHistory() {
    const user_id = localStorage.getItem("user_id");

    if (!user_id) {
        alert("Not logged in!");
        return;
    }

    const res = await fetch(API_URL + "/user/" + user_id + "/history");

    if (!res.ok) {
        alert("Failed to load history");
        return;
    }

    const history = await res.json();
    const tbody = document.getElementById("historyBody");
    tbody.innerHTML = "";

    history.forEach(item => {
        const row = document.createElement("tr");

        const safeAdvice = item.advice
            ? item.advice.replace(/\n/g, "<br>")
            : "No advice stored";

        row.innerHTML = `
            <td>${item.prediction_id}</td>
            <td>${item.risk}</td>
            <td>${item.time}</td>
            <td>${safeAdvice}</td>
            <td><button onclick="deleteOne(${item.prediction_id})">Delete</button></td>
        `;

        tbody.appendChild(row);
    });
}

/* ---------- DELETE SINGLE RECORD ---------- */

async function deleteOne(predictionId) {
    const confirmDelete = confirm("Delete this prediction?");
    if (!confirmDelete) return;

    const res = await fetch(API_URL + "/user/history/" + predictionId, {
        method: "DELETE"
    });

    if (!res.ok) {
        alert("Failed to delete prediction");
        return;
    }

    const data = await res.json();
    alert(data.message);

    loadHistory(); // refresh table
}

/* ---------- CLEAR ALL HISTORY ---------- */

async function clearHistory() {
    const user_id = localStorage.getItem("user_id");

    if (!user_id) {
        alert("Not logged in!");
        return;
    }

    const confirmDelete = confirm("Are you sure you want to delete ALL history?");
    if (!confirmDelete) return;

    const res = await fetch(API_URL + "/user/" + user_id + "/history", {
        method: "DELETE"
    });

    const data = await res.json();
    alert(data.message);

    loadHistory();
}

/* ---------- AUTO LOAD WHEN PAGE OPENS ---------- */
window.onload = loadHistory;
