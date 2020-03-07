import React from 'react';
import {Route, Switch} from 'react-router-dom';

import Home from '../pages/Home';
import SignUp from '../pages/SignUp';
import SignIn from '../pages/SignIn';
import Dashboard from "../pages/Dashboard";

const Main = () => {
  return (
    <Switch>
      <Route exact path='/' component={Home}/>
      <Route exact path='/signup' component={SignUp}/>
      <Route exact path='/signin' component={SignIn}/>
      <Route exact path='/dashboard' component={Dashboard}/>
    </Switch>
  );
};

export default Main;
