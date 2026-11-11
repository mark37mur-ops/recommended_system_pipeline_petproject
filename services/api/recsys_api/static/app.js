"use strict";

const GENRES_RU = {
  "Action": "боевик", "Adventure": "приключения", "Animation": "мультфильм",
  "Children's": "детский", "Comedy": "комедия", "Crime": "криминал",
  "Documentary": "документальный", "Drama": "драма", "Fantasy": "фэнтези",
  "Film-Noir": "нуар", "Horror": "ужасы", "Musical": "мюзикл", "Mystery": "детектив",
  "Romance": "мелодрама", "Sci-Fi": "фантастика", "Thriller": "триллер",
  "War": "военный", "Western": "вестерн",
};
const PATH_LABELS = {
  full: "персонально · ALS + LightGBM",
  cold_start: "популярное",
  fallback: "популярное (модель недоступна)",
};
const MIN = window.MIN_RATINGS || 5;
const myRatings = new Map(); // movie_id -> rating, for the cards on screen
let nRatings = 0;
let refreshTimer = null;

const $ = (id) => document.getElementById(id);

function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 3500);
}

async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${response.status}`);
  return response.json();
}

function setProgress(n) {
  if (typeof n !== "number") return;
  nRatings = n;
  const share = Math.min(n / MIN, 1);
  $("progress-fill").style.width = `${share * 100}%`;
  $("progress-text").textContent = n >= MIN
    ? `Оценено ${n} — рекомендации персональные`
    : `Оценено ${n} из ${MIN}`;
}

function renderStars(container, movie) {
  container.replaceChildren();
  const current = myRatings.get(movie.movie_id) || 0;
  for (let value = 1; value <= 5; value += 1) {
    const star = document.createElement("button");
    star.type = "button";
    star.className = "star" + (value <= current ? " on" : "");
    star.textContent = "★";
    star.setAttribute("role", "radio");
    star.setAttribute("aria-checked", String(value === current));
    star.setAttribute("aria-label", `${value} из 5`);
    star.addEventListener("click", () => rate(movie, value));
    container.append(star);
  }
}

function card(movie) {
  const node = $("card-template").content.firstElementChild.cloneNode(true);
  node.dataset.movieId = movie.movie_id;
  node.querySelector(".title").textContent = movie.title;
  const genres = movie.genres.map((g) => GENRES_RU[g] || g).join(", ");
  node.querySelector(".meta").textContent = [movie.year, genres].filter(Boolean).join(" · ");
  renderStars(node.querySelector(".stars"), movie);
  return node;
}

function renderGrid(gridId, movies, emptyText) {
  const grid = $(gridId);
  for (const m of movies) {
    if (m.user_rating) myRatings.set(m.movie_id, m.user_rating);
  }
  if (!movies.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = emptyText;
    grid.replaceChildren(p);
    return;
  }
  grid.replaceChildren(...movies.map(card));
}

function refreshStars(movieId) {
  document.querySelectorAll(`.card[data-movie-id="${movieId}"] .stars`).forEach((stars) => {
    renderStars(stars, { movie_id: movieId });
  });
}

async function rate(movie, value) {
  const response = await fetch("/ratings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ movie_id: movie.movie_id, rating: value }),
  });
  if (response.status === 429) return toast("Слишком много оценок подряд — подождите минуту.");
  if (response.status === 403) return toast("Достигнут лимит оценок для одного посетителя.");
  if (response.status === 503) return toast("База временно недоступна, попробуйте позже.");
  if (!response.ok) return toast("Не удалось сохранить оценку.");
  const body = await response.json();
  myRatings.set(movie.movie_id, value);
  refreshStars(movie.movie_id);
  setProgress(body.n_ratings);
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(loadRecommendations, 300);
}

async function loadRecommendations() {
  try {
    const recs = await getJSON("/recommendations?k=10");
    setProgress(recs.n_ratings);
    const badge = $("recs-badge");
    badge.textContent = PATH_LABELS[recs.path] || recs.path;
    badge.className = `badge ${recs.path}`;
    if (recs.path === "full" && recs.model_version) {
      badge.textContent += ` · модель v${recs.model_version}`;
    }
    const left = Math.max(MIN - (recs.n_ratings || 0), 0);
    let hint = "";
    if (recs.path === "cold_start" && left > 0) {
      hint = (recs.n_ratings || 0) === 0
        ? `Пока вы ничего не оценили, показываем самые популярные фильмы. Оцените ещё ${left}.`
        : `Пока это популярное в жанрах, которые вам понравились. Оцените ещё ${left}, чтобы включилась модель.`;
    } else if (recs.path === "fallback") {
      hint = "Персональная модель сейчас недоступна, показываем популярные фильмы.";
    }
    $("recs-hint").textContent = hint;
    renderGrid("recs-grid", recs.items, "Нет рекомендаций.");
  } catch (err) {
    $("recs-hint").textContent = "Не удалось получить рекомендации.";
  }
}

async function loadOnboarding() {
  try {
    const data = await getJSON("/movies/onboarding");
    renderGrid("onboarding-grid", data.items, "Каталог фильмов ещё загружается.");
  } catch (err) {
    renderGrid("onboarding-grid", [], "Каталог фильмов недоступен.");
  }
}

function setupSearch() {
  let timer = null;
  let seq = 0;
  $("search").addEventListener("input", (event) => {
    const q = event.target.value.trim();
    clearTimeout(timer);
    if (q.length < 2) {
      $("search-grid").replaceChildren();
      return;
    }
    timer = setTimeout(async () => {
      const mine = ++seq;
      try {
        const data = await getJSON(`/movies/search?q=${encodeURIComponent(q)}`);
        if (mine === seq) renderGrid("search-grid", data.items, "Ничего не нашлось.");
      } catch (err) {
        if (mine === seq) renderGrid("search-grid", [], "Поиск недоступен.");
      }
    }, 250);
  });
}

setupSearch();
loadOnboarding().then(loadRecommendations);
