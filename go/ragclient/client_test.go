package ragclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAskSendsQuestionAndDecodesAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/ask" {
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["question"] != "Which port?" || body["k"].(float64) != 3 {
			t.Fatalf("unexpected body %v", body)
		}
		json.NewEncoder(w).Encode(Answer{Answer: "port 52116 [router#Control Protocol]",
			Citations: []string{"router#Control Protocol"}, Grounded: true})
	}))
	defer srv.Close()

	a, err := New(srv.URL).Ask(context.Background(), "Which port?", 3)
	if err != nil {
		t.Fatal(err)
	}
	if !a.Grounded || len(a.Citations) != 1 || a.Citations[0] != "router#Control Protocol" {
		t.Fatalf("unexpected answer %+v", a)
	}
}

func TestAskReturnsErrorOnHTTPFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad", http.StatusUnprocessableEntity)
	}))
	defer srv.Close()
	if _, err := New(srv.URL).Ask(context.Background(), "x", 3); err == nil {
		t.Fatal("expected an error for HTTP 422")
	}
}
