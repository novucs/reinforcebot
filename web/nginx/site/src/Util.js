export const BASE_URL = 'http://localhost:8080';

export function displayErrors(...errors) {
  return errors.flatMap((e) => displayError(e));
}

export default function displayError(error) {
  if (error == null) return [];
  if (error instanceof Array) return error;
  return [error];
}

export function hasJWT() {
  let jwt = window.localStorage.getItem('jwtAccess');
  return !(jwt === null || jwt === "null");
}

export function getJWT() {
  return window.localStorage.getItem('jwtAccess');
}

export function ensureSignedIn() {
  if (!hasJWT()) {
    window.location = '/signin';
  }
}

export function ensureSignedOut() {
  if (hasJWT()) {
    window.location = '/dashboard';
  }
}

export function signOut() {
  window.localStorage.setItem('jwtAccess', null);
  window.localStorage.setItem('jwtRefresh', null);
  window.location = '/signin';
}

export function signIn(username, password, callback) {
  fetch(BASE_URL + '/api/auth/jwt/create/', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      username: username,
      password: password,
    }),
  }).then(response => {
    if (response.status !== 200) {
      response.json().then(body => {
        callback(displayError(body['detail']));
      });
      return;
    }

    callback([]);
    response.json().then(body => {
      window.localStorage.setItem('jwtAccess', body['access']);
      window.localStorage.setItem('jwtRefresh', body['refresh']);
      window.location.reload();
    });
  });
}

export function refreshJWT() {
  let token = window.localStorage.getItem('jwtRefresh');
  fetch(BASE_URL + '/api/auth/jwt/refresh/', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({refresh: token}),
  }).then(response => {
    if (response.status === 401) {
      signOut();
      return;
    }

    if (response.status !== 200) {
      response.json().then(body => {
        console.error("Unable to refresh JWT: ", body);
      });
      return;
    }

    response.json().then(body => {
      window.localStorage.setItem('jwtAccess', body['access']);
      window.location = '/dashboard';
    });
  });
}
