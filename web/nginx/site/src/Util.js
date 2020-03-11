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
