import React from 'react';
import { Route, IndexRoute } from 'react-router';

import App from './components/App';
import MainPage from './components/MainPage';
import SomePage from './components/SomePage';
import SomeOtherPage from './components/SomeOtherPage';

export default (
  <Route path="/" component={App}>
    <IndexRoute component={MainPage} />
    <Route path="/some/where" component={SomePage} />
    <Route path="/some/otherpage" component={SomeOtherPage} />
  </Route>
);
