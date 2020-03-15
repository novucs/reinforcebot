import React from 'react';
import {Route, Switch} from 'react-router-dom';

import Home from '../pages/Home';
import SignIn from '../pages/SignIn';
import Agents from "../pages/Agents";
import Start from "../pages/Start";
import AgentDetail from "../pages/AgentDetail";
import SignUp from "../pages/SignUp";

export default class Main extends React.Component {
  render = () => (
    <Switch>
      <Route exact path='/' component={Home}/>
      <Route exact path='/start' component={Start}/>
      <Route exact path='/signup' component={SignUp}/>
      <Route exact path='/signin' component={SignIn}/>
      <Route exact path='/agents' component={Agents}/>
      <Route path='/agent/:id' component={AgentDetail}/>
    </Switch>
  );
}
