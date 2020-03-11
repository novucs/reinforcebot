import React, {Component} from "react";
import {Button, Container, Menu} from "semantic-ui-react";
import {hasJWT} from "./Util";

export default class TopMenu extends Component {

  signOut() {
    window.localStorage.setItem('jwtAccess', null);
    window.localStorage.setItem('jwtRefresh', null);
    console.log("called");
    window.location = '/';
  }

  render() {
    return (
      <Menu fixed='top' size='large' inverted>
        <Container>
          <Menu.Item as='a' href='/'>
            Home
          </Menu.Item>
          <Menu.Item as='a' href='/dashboard'>
            Dashboard
          </Menu.Item>
          <Menu.Item position='right'>
            <div hidden={hasJWT()}>
              <Button as='a' href='/signin'>
                Sign in
              </Button>
              <Button as='a' href='/signup' primary style={{marginLeft: '0.5em'}}>
                Sign Up
              </Button>
            </div>
            <div hidden={!hasJWT()}>
              <Button onClick={this.signOut} negative>
                Sign Out
              </Button>
            </div>
          </Menu.Item>
        </Container>
      </Menu>
    )
  }
}
