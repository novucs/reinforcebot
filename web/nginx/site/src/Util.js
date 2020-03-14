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
      signOut();
      response.json().then(body => {
        console.error("Unable to refresh JWT: ", body);
      });
      return;
    }

    response.json().then(body => {
      window.localStorage.setItem('jwtAccess', body['access']);
      window.location.reload();
    });
  });
}


export function fetchUsers(userURIs, callback) {
  if (!hasJWT()) {
    signOut();
    return;
  }

  let userIDs = new Set();
  userURIs.forEach(userURI => {
    let id = userURI.replace(/\/$/, "");
    id = id.substring(id.lastIndexOf('/') + 1);
    userIDs.add(id);
  });

  let users = {};
  userIDs.forEach(userID => {
    let userURI = BASE_URL + '/api/users/' + userID + '/';
    fetch(userURI, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to fetch user (" + userID + "): ", body);
        });
        return;
      }

      response.json().then(body => {
        users[userID] = body;
        users[userURI] = body;
        users[BASE_URL + '/api/auth/users/' + userID + '/'] = body;
        callback(users);
      });
    });
  });
}

export function deleteAgent(id, callback) {
  if (hasJWT()) {
    fetch(BASE_URL + '/api/agents/' + id + '/', {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 204) {
        console.error("Unable to delete agent: ", response);
        return;
      }

      callback();
    });
  }
}

export function createAgent(name, description, parametersFile, callback) {
  let data = new FormData();
  data.append('name', name);
  data.append('description', description);
  data.append('parameters', parametersFile);
  data.append('changeReason', 'Initial creation');

  fetch(BASE_URL + '/api/agents/', {
    method: 'POST',
    headers: {
      'Authorization': 'JWT ' + getJWT(),
    },
    body: data,
  }).then((response) => {
    if (response.status === 401) {
      refreshJWT();
      return;
    }

    if (response.status !== 201) {
      console.error('Failed to create an agent: ', response);
      return;
    }

    callback();
  });
}

export function cropText(text, maxLength) {
  let uncropped = text.split('\n')[0];
  let cropped = uncropped.substring(0, maxLength);
  if (uncropped.length > maxLength) {
    return cropped + '...';
  }
  return cropped;
}
